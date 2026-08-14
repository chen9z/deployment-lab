# PVE Disk Migration Runbook

This runbook records the observed Windows/WSL disk layout and the agreed PVE
storage design. It is intended to be opened from a future Codex session on the
new PVE host and executed one phase at a time.

## Safety contract for the future operator/agent

- Treat every `wipefs`, `sgdisk`, `pvcreate`, `mkfs`, and PVE installer action
  as destructive.
- Never identify a destructive target only by `/dev/sdX`, `/dev/nvmeXnY`, a
  drive letter, capacity, or device order. Match model, serial, and stable
  `/dev/disk/by-id/` path together.
- Before each destructive phase, print `lsblk`, `blkid`, `wipefs -n`, and the
  resolved `/dev/disk/by-id/` symlink, then obtain explicit human confirmation.
- Do not destroy E until the copied VHDX on F has the same SHA-256 hash as the
  source. Do not destroy F until the recovered files on E have been inspected
  and the important models, projects, private checkpoints, and F's existing
  files have been verified.
- Do not mount a hibernated or dirty Windows NTFS filesystem read-write.
- Do not reuse commands here blindly if a model or serial differs from the
  inventory below. Stop and investigate the mismatch.
- Preserve unrelated user changes in the repository and on every disk.

## Observed inventory before migration

Inventory date: 2026-08-13/15, from Windows PowerShell and WSL.

| Windows disk | Current letter | Hardware | Size | Serial | Current role |
|---|---|---|---:|---|---|
| Disk 1 | C | SK hynix Platinum P41 1TB NVMe (`SHPP41-1000GM`) | 931.5 GiB | `ACE4_2E00_35E2_8B59_2EE4_AC00_0000_0001` | Windows EFI, C, recovery; preserve completely |
| Disk 3 | D | Samsung 970 EVO Plus 500GB NVMe | 465.8 GiB | `0025_3855_21B0_2D78` | Nearly empty; destructive PVE install target |
| Disk 2 | E | SK hynix Platinum P41 2TB NVMe (`SHPP41-2000GM`) | 1863 GiB | `ACE4_2E00_35FE_47DA_2EE4_AC00_0000_0001` | WSL VHDX and Windows pagefile; future PVE fast pool |
| Disk 0 | F | Intel/Solidigm D3-S4520 1.92TB SATA (`SSDSC2KB019TZ`) | 1788.5 GiB | `BTYI137103BX1P9DGN` | About 12 GiB used before migration staging; future shared ext4 store |

Important identification details:

- C and E are both SK hynix P41 drives. The 1TB Windows disk serial contains
  `8B59`; the 2TB future fast-pool disk serial contains `47DA`.
- D is the only Samsung 970 EVO Plus 500GB and is the only intended PVE
  installer target.
- F is the 2.5-inch enterprise SATA SSD. It is not empty.
- Linux device numbering will differ after PVE is installed.

## Observed data that must survive

The WSL distribution is registered at:

```text
E:\Application\WSL\Ubuntu-22.04\ext4.vhdx
```

Observed physical VHDX file size:

```text
1399.79 GiB
```

The WSL filesystem had about 1.4 TiB used under `/home/looper`. Large observed
areas included:

| Path | Approximate size |
|---|---:|
| `/home/looper/workspace/deployment-lab/models` | 217 GiB |
| `/home/looper/.cache/huggingface` | 246 GiB |
| `/home/looper/workspace/ComfyUI` | 231 GiB |
| `/home/looper/workspace/llama.cpp` | 185 GiB |
| `/home/looper/.cache/llama.cpp` | 134 GiB |
| `/home/looper/.cache/modelscope` | 49 GiB |

Copying only `deployment-lab/models` is therefore insufficient. Preserve the
whole VHDX for the first migration.

E also contained a 64 GiB Windows `pagefile.sys`; it does not need migration.

F used about 12 GiB on the latest 2026-08-15 check, before the future migration
copy. Observed top-level entries included:

```text
F:\EFI
F:\quark
```

Inspect and preserve these before formatting F. The top-level `EFI` directory
was on a normal basic-data partition, not a Windows system partition, but its
contents must still be reviewed before deletion.

C also contains Docker Desktop data, including an observed file around 353 GiB:

```text
C:\Users\chen\AppData\Local\Docker\wsl\disk\docker_data.vhdx
```

C remains untouched. This Docker VHDX is not required to reconstruct the model
deployment if the Compose files and model data survive.

## Agreed final storage design

```text
Samsung 970 EVO Plus 500GB (former D)
└── PVE system disk
    ├── PVE root/local
    └── local-lvm for VM operating-system disks

SK hynix P41 2TB (former E)
└── PVE-managed LVM-thin pool: fast-nvme
    ├── GPU VM hot/training virtual disk
    ├── Agent/dev VM workspace virtual disks
    └── future high-I/O VM disks

Intel/Solidigm D3-S4520 1.92TB (former F)
└── Host-mounted ext4: /data
    ├── models (published/shared weights)
    ├── datasets/raw
    ├── checkpoints/<vm-or-job>
    ├── outputs
    ├── archive
    └── backups

SK hynix P41 1TB (C)
└── Existing Windows installation; never add to a PVE storage pool
```

E is managed by PVE rather than passed through whole because multiple VMs may
need fast virtual disks. F is mounted once by the PVE host and later shared to
Linux guests with VirtioFS. Never attach the same ext4 block filesystem to
multiple VMs concurrently.

## Phase 0: Windows preflight and staging

Run this phase before booting the PVE installer.

### 0.1 Preserve encryption and boot information

- Save all BitLocker recovery keys.
- Confirm whether C, E, or F is BitLocker-encrypted.
- Confirm the Windows account can read all files on E and F.
- Export or photograph the Windows Disk Management layout.

Read-only inventory commands in an elevated PowerShell:

```powershell
Get-PhysicalDisk |
  Sort-Object DeviceId |
  Format-Table DeviceId,FriendlyName,SerialNumber,MediaType,BusType,
    @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}},HealthStatus -AutoSize

Get-Disk |
  Sort-Object Number |
  Format-Table Number,FriendlyName,SerialNumber,BusType,
    @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}},
    PartitionStyle,IsBoot,IsSystem,HealthStatus -AutoSize

Get-Partition |
  Sort-Object DiskNumber,Offset |
  Format-Table DiskNumber,PartitionNumber,DriveLetter,
    @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}},Type,IsBoot,IsSystem -AutoSize

Get-Volume |
  Sort-Object DriveLetter |
  Format-Table DriveLetter,FileSystemLabel,FileSystem,
    @{N='SizeGB';E={[math]::Round($_.Size/1GB,1)}},
    @{N='FreeGB';E={[math]::Round($_.SizeRemaining/1GB,1)}},HealthStatus -AutoSize
```

Expected mapping must match the inventory table above.

### 0.2 Stop Windows hibernation, Docker, and WSL

In an elevated PowerShell:

```powershell
powercfg /h off
```

Quit Docker Desktop completely, stop any remaining WSL workloads, and then:

```powershell
wsl --shutdown
wsl --list --verbose
```

All distributions should show `Stopped`. Do not restart Docker or WSL until the
VHDX copy and hash verification finish.

### 0.3 Copy the WSL VHDX from E to F

F had enough free space at inspection time (about 1.67 TiB) for the 1.4 TiB
VHDX. Recheck free space before copying.

```powershell
New-Item -ItemType Directory F:\PVE-Migration -Force

robocopy `
  E:\Application\WSL\Ubuntu-22.04 `
  F:\PVE-Migration `
  ext4.vhdx `
  /J /R:2 /W:5 /COPY:DAT
```

`robocopy` exit codes below 8 are not necessarily failures; inspect its final
summary and the destination file size.

### 0.4 Hash both VHDX files

This reads about 2.8 TiB in total and may take hours.

```powershell
Get-FileHash `
  E:\Application\WSL\Ubuntu-22.04\ext4.vhdx `
  -Algorithm SHA256

Get-FileHash `
  F:\PVE-Migration\ext4.vhdx `
  -Algorithm SHA256
```

Record the hash here before installation:

```text
SHA256: _________________________________________________
Source length: _________________________________________
Destination length: ____________________________________
Verification date: _____________________________________
```

Stop if the hashes differ.

### 0.5 Preserve small, irreplaceable data separately

The full VHDX is the migration copy, not a robust backup. Also copy the most
valuable small data to another independent destination if available:

- private fine-tunes, LoRA adapters, and checkpoints;
- uncommitted Git work and repository configuration;
- SSH public configuration (not private keys unless encrypted appropriately);
- Compose files, chat templates, benchmark results worth retaining;
- private datasets that cannot be downloaded again.

### 0.6 Shut Windows down cleanly

Use a full shutdown, not sleep or hibernation:

```powershell
shutdown /s /t 0
```

## Phase 1: Install PVE on D only

The safest procedure is to disconnect or disable C, E, and F in firmware or
physically, leaving only the Samsung 970 EVO Plus 500GB connected.

PVE installer target must match all of:

```text
Model: Samsung SSD 970 EVO Plus 500GB
Serial: 0025_3855_21B0_2D78
Capacity: approximately 465.8 GiB
Former Windows letter: D
```

Install current stable PVE to the whole Samsung disk. Do not select either SK
hynix P41 or the Intel/Solidigm SATA SSD.

After installation:

1. Boot PVE successfully with only the Samsung disk.
2. Configure a static management address, hostname, DNS, NTP, and repositories.
3. Update PVE and reboot once before adding data disks.
4. Reconnect E and F.
5. C may remain disconnected until the other disks are positively identified.

## Phase 2: Inventory disks on PVE

This phase is read-only.

```bash
lsblk -e7 -o NAME,PATH,TYPE,SIZE,FSTYPE,FSVER,LABEL,PARTLABEL,UUID,MOUNTPOINTS,MODEL,SERIAL,TRAN,ROTA
ls -l /dev/disk/by-id/
blkid
pvs
vgs
lvs -a -o +devices
pvesm status
```

Install health tools if needed:

```bash
apt update
apt install --yes smartmontools nvme-cli qemu-utils gdisk parted
```

Inspect all candidate disks without changing them:

```bash
smartctl --scan-open
nvme list
```

Then run `smartctl -a` or `nvme smart-log` against each resolved device. Record
the PVE `by-id` paths here:

```text
C / Windows P41 1TB:
  /dev/disk/by-id/________________________________________

D / PVE Samsung 500GB:
  /dev/disk/by-id/________________________________________

E / future fast P41 2TB:
  /dev/disk/by-id/________________________________________

F / staging/future shared Intel 1.92TB:
  /dev/disk/by-id/________________________________________
```

Expected E identity contains model `SHPP41-2000GM` and serial component
`35FE_47DA`. Expected F identity contains model `SSDSC2KB019TZ` and serial
`BTYI137103BX1P9DGN`.

Before any destructive operation, also run against the exact stable path:

```bash
readlink -f /dev/disk/by-id/<candidate>
udevadm info --query=property --name=/dev/disk/by-id/<candidate>
wipefs --no-act /dev/disk/by-id/<candidate>
```

## Phase 3: Create the PVE fast pool on E

### Destructive checkpoint E

Do not continue until all statements are true:

- F contains `F:\PVE-Migration\ext4.vhdx`.
- Its SHA-256 matched the original E VHDX before PVE installation.
- The target is the 2TB `SHPP41-2000GM`, serial containing `35FE_47DA`.
- The target is not the 1TB Windows P41, Samsung PVE disk, or Intel staging disk.
- The human explicitly authorizes destroying E.

The commands below intentionally retain a placeholder. Replace it only after
the checks above; do not define a broad or guessed shell variable.

```bash
wipefs --all /dev/disk/by-id/<EXACT_E_P41_2TB_BY_ID>
sgdisk --zap-all /dev/disk/by-id/<EXACT_E_P41_2TB_BY_ID>
partprobe /dev/disk/by-id/<EXACT_E_P41_2TB_BY_ID>

pvcreate /dev/disk/by-id/<EXACT_E_P41_2TB_BY_ID>
vgcreate fast-vg /dev/disk/by-id/<EXACT_E_P41_2TB_BY_ID>
lvcreate --type thin-pool --extents 95%FREE --name fast-thin fast-vg

pvesm add lvmthin fast-nvme \
  --vgname fast-vg \
  --thinpool fast-thin \
  --content images,rootdir
```

Verify:

```bash
pvs
vgs
lvs -a -o +devices,data_percent,metadata_percent
pvesm status
```

Expected PVE storage ID: `fast-nvme`.

Do not over-provision this thin pool casually. Configure monitoring and keep
actual data usage below about 80-85%; a full thin pool can corrupt guest
filesystems.

## Phase 4: Build a temporary recovery-capable Ubuntu VM

The exact VM ID is intentionally not fixed here. Before creation, inspect:

```bash
pvesh get /cluster/nextid
qm list
```

Create the main GPU Ubuntu VM, or a temporary recovery Ubuntu VM, with:

- OS disk on D's `local-lvm` (roughly 100-160 GiB);
- VirtIO SCSI single controller;
- a large raw virtual data disk on `fast-nvme`;
- discard, SSD emulation, and I/O thread enabled for the data disk;
- enough virtual capacity to recover `/home/looper` before pruning caches.

Because the source used about 1.4 TiB, size the initial E-backed virtual data
disk conservatively and monitor physical thin-pool usage. A nominal thin disk
around 1.6-1.7 TiB can be used for recovery, but do not create other large
thin disks until old caches are pruned and actual usage is known.

Install Ubuntu and format only the newly created E-backed virtual data disk as
ext4. Mount it inside the guest at:

```text
/workspace
```

Do not format a disk until its virtual-disk identity is confirmed inside the
guest using `lsblk -o NAME,SIZE,MODEL,SERIAL,FSTYPE,MOUNTPOINTS`.

## Phase 5: Temporarily attach F to the recovery VM

F must remain NTFS during recovery because it holds the VHDX and its
pre-existing files.

1. Shut the recovery VM down.
2. Resolve the exact F `by-id` path using the Intel model and serial.
3. Attach that physical disk only to the recovery VM using a stable `by-id`
   path. Do not attach it to two running guests.
4. Boot the VM and mount F read-only first.

The precise `qm set` slot depends on the VM's existing hardware. Inspect
`qm config <VMID>` before adding anything. A future agent must choose an unused
SCSI/SATA slot and show the resolved path before applying it.

Inside Ubuntu, inspect:

```bash
lsblk -o NAME,PATH,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS,MODEL,SERIAL
sudo blkid
```

Create a mount point and mount the NTFS partition read-only. Replace the
placeholder with the identified F partition, not the whole disk:

```bash
sudo mkdir -p /mnt/migration-f
sudo mount -t ntfs3 -o ro /dev/<F_NTFS_PARTITION> /mnt/migration-f
findmnt /mnt/migration-f
```

If NTFS reports hibernation, dirty state, or an unsupported feature, stop. Do
not use a force or remove-hibernation option against the only migration copy.

## Phase 6: Mount the copied WSL VHDX read-only

Inside the recovery VM:

```bash
sudo apt update
sudo apt install --yes qemu-utils rsync
sudo modprobe nbd max_part=8

sudo qemu-nbd \
  --read-only \
  --connect=/dev/nbd0 \
  /mnt/migration-f/PVE-Migration/ext4.vhdx

lsblk -f /dev/nbd0
sudo blkid /dev/nbd0 /dev/nbd0p1 2>/dev/null || true
```

A WSL VHDX commonly exposes ext4 directly as `/dev/nbd0`, but it may expose a
partition. Use `lsblk`/`blkid`; never guess. Mount the discovered ext4 device
read-only without journal replay:

```bash
sudo mkdir -p /mnt/old-wsl
sudo mount -o ro,noload /dev/<NBD_EXT4_DEVICE> /mnt/old-wsl
findmnt /mnt/old-wsl
```

Confirm expected content before copying:

```bash
sudo ls -la /mnt/old-wsl/home/looper
sudo du -x -h --max-depth=1 /mnt/old-wsl/home/looper | sort -h
df -h /workspace
```

## Phase 7: Recover WSL and existing F files onto E

Keep the source mounts read-only. Copy with metadata and resumability:

```bash
sudo mkdir -p /workspace/recovered/home-looper

sudo rsync -aHAXS --numeric-ids --info=progress2 --partial \
  /mnt/old-wsl/home/looper/ \
  /workspace/recovered/home-looper/
```

Review F's existing top-level directories, then copy them into a clearly
separate recovery area before F is formatted:

```bash
sudo mkdir -p /workspace/recovered/former-f
sudo rsync -aH --info=progress2 --partial \
  --exclude='PVE-Migration/ext4.vhdx' \
  /mnt/migration-f/ \
  /workspace/recovered/former-f/
```

The NTFS source cannot represent all Linux ownership/xattr semantics, but the
VHDX-to-ext4 copy can. Preserve `-aHAXS --numeric-ids` for the WSL copy.

Verification checklist before destroying F:

- `deployment-lab` is present and `git status` is sensible.
- Private/uncommitted work is present.
- Model directories contain expected shard counts and sizes.
- ComfyUI, llama.cpp, Hugging Face, and ModelScope data are present.
- Private adapters/checkpoints open successfully.
- The existing files from F have been reviewed under `former-f`.
- A second copy exists for irreplaceable private data.
- At least a sample of large model files has been hashed on source and
  destination, or an `rsync --checksum --dry-run` verification has been run for
  the irreplaceable subsets.

Full checksum verification is expensive but can be run for selected trees:

```bash
sudo rsync -aHAXS --numeric-ids --checksum --dry-run --itemize-changes \
  /mnt/old-wsl/home/looper/workspace/deployment-lab/ \
  /workspace/recovered/home-looper/workspace/deployment-lab/
```

Unmount cleanly when finished:

```bash
sudo umount /mnt/old-wsl
sudo qemu-nbd --disconnect /dev/nbd0
sudo umount /mnt/migration-f
```

Shut down the VM and detach the physical F disk from its PVE configuration.
Verify `qm config <VMID>` no longer references F before proceeding.

## Phase 8: Format F as the shared ext4 data store

### Destructive checkpoint F

Do not continue until all statements are true:

- Recovery from the VHDX onto E completed and was verified.
- F's pre-existing files were copied and reviewed.
- Important private data has another independent copy.
- The F physical disk is detached from every VM and unmounted everywhere.
- The target model is `SSDSC2KB019TZ`, serial `BTYI137103BX1P9DGN`.
- The human explicitly authorizes destroying F, including the only staging
  VHDX copy on that disk.

On the PVE host, inspect one last time:

```bash
lsblk -e7 -o NAME,PATH,SIZE,FSTYPE,LABEL,UUID,MOUNTPOINTS,MODEL,SERIAL
readlink -f /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
udevadm info --query=property --name=/dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
wipefs --no-act /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
```

After explicit confirmation:

```bash
wipefs --all /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
sgdisk --zap-all /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
sgdisk --new=1:0:0 --typecode=1:8300 --change-name=1:pve-data \
  /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
partprobe /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>
udevadm settle

mkfs.ext4 -L pve-data -m 0 \
  /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>-part1
```

Verify the `-part1` symlink exists before `mkfs.ext4`; SATA `by-id` naming is
expected to expose it, but stop if it does not.

Create the host mount point:

```bash
mkdir -p /data
blkid /dev/disk/by-id/<EXACT_F_INTEL_BY_ID>-part1
```

Add an `/etc/fstab` entry using the new filesystem UUID, not `/dev/sdX`:

```text
UUID=<NEW_F_EXT4_UUID> /data ext4 defaults,noatime 0 2
```

Do not add `nofail`: silently starting guests without the shared disk can cause
writes to land on D under the empty mount-point directory.

Test:

```bash
mount -a
findmnt /data
df -hT /data
touch /data/.mount-test
rm /data/.mount-test
systemctl enable --now fstrim.timer
```

Create the initial hierarchy:

```bash
mkdir -p \
  /data/models \
  /data/datasets/raw \
  /data/checkpoints \
  /data/outputs \
  /data/archive \
  /data/backups
```

Do not expose the entire root read-write to every VM. Publish model and raw
dataset directories read-only, and give each VM/job its own writable
checkpoint/output directory.

## Phase 9: Publish F to VMs with VirtioFS

PVE 8.4+ supports VirtioFS directory mappings. Use the current PVE GUI under
Datacenter resource mappings, or inspect the current CLI schema before adding
mappings:

```bash
pvesh help create /cluster/mapping/dir --verbose
qm help set | sed -n '/virtiofs/,+20p'
```

Suggested mappings:

```text
models-ro       -> a host read-only bind view of /data/models
datasets-ro     -> a host read-only bind view of /data/datasets/raw
gpu-checkpoints -> /data/checkpoints/gpu-vm
agent-outputs   -> /data/outputs/agent-vm
```

Codex/operator must verify the installed PVE command syntax before applying.
Do not infer a read-only guarantee merely from guest mount options; enforce it
at the host export/bind and permission layer.

Inside Linux guests, mount using the configured VirtioFS tag, for example:

```bash
sudo mkdir -p /models
sudo mount -t virtiofs models-ro /models
```

VirtioFS content is outside normal VM disk snapshots/backups. Back up private
checkpoints and datasets separately.

## Phase 10: Reorganize recovered data

After F is available to the GPU VM, use this policy:

```text
E-backed VM disk /workspace:
  active models
  processed/tokenized datasets
  current training workspace
  scratch and NVMe offload
  Torch/Triton compile caches
  current one or two checkpoints

F-backed host /data, selectively mapped into guests:
  published/cold model weights
  raw datasets
  historical checkpoints
  final outputs
  downloadable caches and archives
```

Do not immediately delete recovered source trees. First start the retained
Compose services and run the repository checks:

```bash
eval/verify.sh
eval/verify-full.sh
eval/bench.sh
```

Use `eval/verify-stress.sh` only after basic verification passes.

Prune only clearly reproducible caches after recording their purpose. Keep at
least 15-20% actual free space on E and 10% on F.

## Phase 11: Windows disk policy

If C is reconnected to PVE:

- Keep it out of all PVE storage pools.
- Do not auto-mount it read-write.
- Do not add it to LVM, ZFS, Ceph, or a VM without a separate reviewed plan.
- Select Windows or PVE through firmware boot selection if dual boot is needed.
- Never allow both Windows bare metal and a VM to write the Windows disk
  concurrently.

## Post-migration record

Fill this section after completion.

```text
PVE version:
PVE hostname:
PVE management address:

PVE Samsung by-id:
Windows C by-id:
E P41 by-id:
F Intel by-id:

fast-nvme VG/thin pool:
F filesystem UUID:
F mount point:

GPU VM ID:
Agent VM ID:
VirtioFS mapping IDs:

Recovery verification date:
Important-data backup location:
Known follow-up work:
```

## Recovery stop conditions

Stop and request human direction if any of these occur:

- A serial/model does not match this runbook.
- The copied VHDX hash was never recorded or differs.
- F free space is insufficient for staging.
- NTFS reports hibernation or corruption.
- `qemu-nbd` does not expose an identifiable ext4 filesystem.
- E does not have enough actual free capacity for recovery.
- The LVM-thin pool approaches full during copy.
- A disk is still mounted or attached to a running VM before a destructive
  command.
- Important private data has only one remaining copy.
- Any proposed operation targets C, the PVE Samsung system disk, a PVE root
  LV, or an unresolved device placeholder.

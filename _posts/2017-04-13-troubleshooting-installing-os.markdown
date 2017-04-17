---
title: Troubleshooting Installing OS
layout: post
published: true
category: [GRUB, tools, OS]
---

Installing OS is a bad job, rescuing one is worse, recovering data is the worst. But most time when it comes, we have to face it.

When you update your OS or install another one, you are taking risks of failing next startup, messing up partitions or even losing valuable data.

Here I record my regular process of installing OSes.

1. Backup: Often it takes a long while but it's worth it.
1. Partition and format a Flash Disk or an internal disk partition. Your should reserve 64 sectors for GRUB2 image Flash Disk (TODO: details). Then format it to FAT32/exFAT, this is for compatibility of GRUB2, if your GRUB2 has embedded NTFS mod or other support, you can format your Flash Disk to corresponding filesystem types. [^1]
1. Install GRUB2 to the boot sectors of your Flash Disk or internal disk, usually you can do it easily in Linux with `grub-install` command, or in Windows with this powerful tool called **Bootice**.
1. Copy the directories `boot` and `grub` into your Flash Disk or internal partition. `boot` is for GRUB2 images, mods and configurations, `grub` is for GRUB4DOS, I just keep GRUB4DOS in case and GRUB2 is enough in theory.
1. Also carry your handy tools with you, I have a [list](./tools_you_need.md) here for you.

# Install macOS

I found I've never been in a situation where I have to install a macOS, I think that's one reason I like macOS better.

# Install Linux

GRUB2 is able to boot Linux iso file using MEMDISK of Syslinux, but often ends up with initramfs error. So you'd better Google the Linux distribution and figure out the location of its kernel file and initrd file, then use GRUB2 `loopback, linux and initrd` commands to mount and boot that iso. The rest installing is too easy.

# Install Windows

## If you want to "Upgrade" your Windows and keep your personal files

If you want to keep files in like `Desktop, Documents, Pictures ... and many settings`, you have to guarantee you can log in old Windows, then mount the Windows Installer iso file and execute the `setup.exe`. If you can't log in the old Windows, then I don't think you can "upgrade", at least I tried to "upgrade" Windows 7 to Windows 10 in `PE.iso->setup.exe` and `extract Windows 10 iso and boot those extract files with BOOTMGR` methods, neither of them worked, Windows 10 installer always told me I must execute "upgrade" in a old running Windows.

## If you don't care rewrite the old Windows partition entirely

Then you have many options:

* Enter PE.iso, write the wim file extracted from iso. (noqa for Windows 10)
* Extract Windows iso file to the root directory of certain volume, then make sure your disk will use BOOTMGR to starup and it can find the `bootmgr` file in the above volume[^2].
* Burn a DVD and install.
* If you have backed up the image of your system partition, just restore it.
* But it seems that you can't just start from GRUB2 on disk and `linux16 /boot/memdisk iso raw; initrd16 ${iso_path}`, it just failed with flashing screen, not sure why.

# Random Thoughts

Windows is always a trouble maker, if a Windows 10 shuts down, it lock this disk in hibernation mode, causing other OSes suffer, you must "restart" Windows 10, and make the "restart" process stop when it's trying to restart, or forcibly shut it. If Windows is installed after Linux, it will make Linux unbootable, but Linux always has Windows in mind. 

[^1]: I need emphasize this again, when you install GRUB2 to internal disk, also remember to have a FAT32/exFAT partition to hold GRUB2 images, mods and configurations, this is for compatibility of GRUB2, if your GRUB2 has embedded NTFS mod or other support, you can format that partition to corresponding filesystem types.

[^2]: When you use this method, you should guarantee your system will be started by the disk you will install Windows into, you can't start from your removable disk but install Windows into a internal disk. To guarantee this, there are some ways: 1. Write MBR to disk's first sector, with partition containing iso extracts set as primary and active, write BOOTMGR bootloader to partition's first sector, restart from disk.
2. Write GRUB2 or GRUB4DOS to disk's first sector, configure GRUB to use `ntldr ${installer_partition}/bootmgr` to boot the Windows installer.
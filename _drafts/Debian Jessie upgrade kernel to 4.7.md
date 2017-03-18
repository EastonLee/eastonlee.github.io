Debian Jessie upgrade kernel to 4.7

wget kernel.ubuntu.com/~kernel-ppa/mainline/v4.7.7/linux-headers-4.7.7-040707_4.7.7-040707.201610071031_all.deb
wget kernel.ubuntu.com/~kernel-ppa/mainline/v4.7.7/linux-headers-4.7.7-040707-generic_4.7.7-040707.201610071031_amd64.deb
wget kernel.ubuntu.com/~kernel-ppa/mainline/v4.7.7/linux-image-4.7.7-040707-generic_4.7.7-040707.201610071031_amd64.deb

sudo dpkg -i linux-headers-4.7.7*.deb linux-image-4.7.7*.deb

vim /etc/default/grub
# first line
# GRUB_DEFAULT='Advanced options for Debian GNU/Linux>Debian GNU/Linux, with Linux 4.7.7-040707-generic'
# GRUB_DEFAULT='Advanced options for Debian GNU/Linux>Debian GNU/Linux, with Linux 3.16.0-4-amd64'

sudo update-grub
# sudo grub-set-default doesn't work seemingly.
sudo reboot
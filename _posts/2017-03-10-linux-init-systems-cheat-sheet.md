---
title: Linux Init Systems Cheat Sheet
layout: post
category: [Linux, tools]
---

Easy way to tell which Init System you are using? And how to manipulate services in different Init System.

# Tell the difference of Init Systems

systemd, SysVinit, Upstart, Supervisor

You must have heard these terms and known their job is "start other process", but given a Linux system, can you tell which Init System it is using? And how to stop or disable service?

init is the first process run on Linux so it has pid=1. init keeps running as long as the system does. All other processes is started by init.

Earlier Linux distributions employed various Init System, but most latest distributions move to systemd.


|   init   |                   platfor                   |          controller          |                                                             directory                                                              |                                             note                                            |
| -------- | ------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| systemd  | Red Hat family, Debian >= 8, Ubuntu >= 15.4 | systemctl                    | /etc/systemd/system/, /run/systemd/system/, /run/systemd/generator.late/, /usr/local/lib/systemd/system/, /usr/lib/systemd/system/ | compatible with SysVinit by systemd-sysv-generator, doesn't honor priorities by /etc/rc?.d/ |
| SysVinit | earlier Linux                               | service                      | /etc/rc/, /etc/init.d/                                                                                                             |                                                                                             |
| Upstart  | Ubuntu < 15.4                               | start, stop, restart, status | /etc/init/                                                                                                                         |                                                                                             |

# How to manipulate services?

|   init   |                              create                              |                                              remove                                              |        start        |        stop        |           enable          |                       disable                        |
| -------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------- | ------------------ | ------------------------- | ---------------------------------------------------- |
| systemd  | create unit_name.type_extension in /etc/systemd/system/          | rm /etc/systemd/system/unit_name.type_extension; systemctl daemon-reload; systemctl reset-failed | systemctl start foo | systemctl stop foo | systemctl enable foo      | systemctl disable foo                                |
| SysVinit | create scripts in /etc/init.d, then run update-rc.d or chkconfig | mv /etc/init.d/foo.conf /etc/init/foo.conf.disabled                                              | service foo start   | service foo stop   | update-rc.d foo enable    | update-rc.d foo disable                              |
| Upstart  | create /etc/init/foo.conf                                        | mv /etc/init/foo.conf /etc/init/foo.conf.disabled                                                | service foo start   | service foo stop   | rm /etc/init/foo.override | echo 'manual' &#124; sudo tee /etc/init/foo.override |

systemctl daemon-reload; systemctl reset-failed


# Supervisor looks simple, can it replace other Init System?

Brief answer: NO. Supervisor's biggest advance is its convenience, but it just covers a subset features which can be done better by like systemd or SystemVinit.

**Reference**

https://unix.stackexchange.com/questions/233468/how-does-systemd-use-etc-init-d-scripts
https://www.turnkeylinux.org/blog/debugging-systemd-sysv-init-compat
http://linoxide.com/linux-command/systemd-vs-sysvinit-cheatsheet/
http://www.tecmint.com/systemd-replaces-init-in-linux/
http://www.pcworld.com/article/2841873/meet-systemd-the-controversial-project-taking-over-a-linux-distro-near-you.html
http://www.tecmint.com/best-linux-init-systems/
https://askubuntu.com/questions/19320/how-to-enable-or-disable-services
https://fedoraproject.org/wiki/SysVinit_to_Systemd_Cheatsheet
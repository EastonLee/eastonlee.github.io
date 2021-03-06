---
title: Why you should not use Ceph S3 for online services
layout: post
published: false
last_modified_at: 2020-02-16
image: https://ceph.com/wp-content/uploads/2016/07/Ceph_Logo_Stacked_RGB_120411_fa.png
category: [distributed]
---

Next time when you use Amazon S3 service, don't take it for granted, because that's very successful engineering work, compared with who? Ceph. Now I'm talking about Ceph S3 (CephFS is another topic that will not be covered here).

<!--more-->

Every time you add OSDs, change their weight, or some disks fail the CRUSH map changes. When CURSH map changes, the cluster gets into recovery mode, PGs are moving around, some PGs are degraded or not available, the objects in those PGs can not be written or even read, if the moving PG is the index, things are much worse, then wait patiently before it recovers. If you are not experienced with Ceph, you can't even know how long the recovery will take.

Ceph is still an amazing software, the core concept is beautiful, but the engineering part is done terribly, there are too many points to improve, I just name some of them. The data distribution between PGs or OSDs is very unbalanced, can cause disk space waste (it has auto balancer now, don't use it, always causing troubles), the commands you can use are too shabby, your maintenance is repetitive and time-consuming, so the auxiliary tool development is your homework (I guess your boss will not appreciate your homework because he/she doesn't even notice you need to build your own tools to run Ceph, am I saying Ceph is lame out of box?), Ceph doesn't give you helpful diagnostic information when something is wrong, it only complains "slow requests", "mon node takes too much disk space", "mds falls behind", "OSD flushing stupidalloc", but what's really happening, why, how to fix? I bet StackOverflow can't help you, the mail list may help sometimes, but most times you just tried the magic out (your boss will doubt if he/she hired a monkey and ask you why not Google it). The resharding is a pain in the neck, if one bucket has more than 100,000 objects, it will split the index, sounds not a big deal right? No, it's a disaster, the bucket can not be read or written, all other buckets performance will be affected badly, many requests will get timeout, and as your buckets grow bigger, the resharding will take longer and longer. Of course you can reshard all buckets ahead of time, but what if your cluster is already running and your users don't allow downtime? So actually Ceph S3 can not even do the reshard in an acceptable way, what an engineering failure. Another point I want to complain about Ceph S3 is, the multi-site configuration never works correctly, in my case after syncing for a while, the sync speed will drop to several KB/s, even much lower than the write speed, then the sync can never finish, totally nonsense, anyway this is not a core feature of Ceph S3 and we can sync S3 data in other ways.

Think about CAP theorem: it is impossible for a distributed data store to simultaneously provide more than two out of the following three guarantees, Consistency, Availability, Partition tolerance. How many guarantees do you think Ceph provides? I think it only provides consistency and partition tolerance at most. Availability is the last thing Ceph guarantees, Ceph gives more priority to the replication and data correctness over the availability. Partition tolerance, I'm not sure if you can write to an object when the OSDs are isolated from each other.

If you use Ceph S3 for offline service, go ahead and continue, weekly daily maintenance doesn't hurt, Ceph can recover to the best working status itself. But if you want to use it for online service, think twice, your users need never stopping services, all the recovery/maintenance/operation should be transparent to users not be felt at all, which Ceph S3 can't guarantee, if you don't want complains about downtime, choose other solutions.

So I don't think Ceph S3 is a production-ready good choice when your services can't endure downtime. Go ahead if your Ceph is for offline services or your users are very nice to wait or you have a bunch of Ceph devs or you are CERN (the future of human relies on Ceph now, good luck human (luckily they mainly use RBD, CephFS not S3)).

Update:

Disable iptalbes in the Ceph cluster, this will save you some trouble.

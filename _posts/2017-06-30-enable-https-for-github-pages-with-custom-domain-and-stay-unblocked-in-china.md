---
title: Enable HTTPS for GitHub Pages with custom domain, and stay unblocked in China
layout: post
published: True 
category: [GitHub, Cloudfront, Cloudflare, Route 53, EC2, AWS, China, Censorship]
---

GitHub Pages is a great place to host static content for personal use, especially with the help of Jekyll, and it's accessible in China. GitHub Pages also supports custom domain, and HTTPS protocol, but not simultaneously, if you configured your custom domain for your GitHub Pages, you can only use HTTP. Maybe one day GitHub will let you HTTPS your own domain on it, but for now you have to get around it.

<!--more-->

# CDNs

It's very easy to think of CDN services which distribute your content throughout the globe and meanwhile support HTTPS, such as Cloundflare and Cloudfront, etc. But Cloudflare and Cloudfront are both blocked in China, at least hardly visitable.

Now I realize it's a headache to ensure my GitHub-hosting site has HTTPS and accessible in China at the same time. Of course you can build your own web server like on EC2 nodes in Japan which are accessible in China and enable its HTTPS, that's a complex but sensible way, but that method will not be covered in this article. In fact I recommend you to host the dynamic part of your site in EC2.

# Geo DNSes

Then I think it would be perfect if only my site is resolved by DNS directly to GitHub server in China, and resolved to CDN server when out of China. Is this technically possible? Definitely YES, and AWS Route 53 comes to help. I set the GitHub Pages server's A record 192.30.252.153 for the China Geolocation, and my Cloudfront's Alias record for the default Geolocation, that's it. Now when Chinese users open http://eastonlee.com, they will be served by GitHub server through HTTP, when other users open http://eastonlee.com, they will be redirected to https://eastonlee.com and served by Cloudfront server through HTTPS.

# Why your site should be global including China?

China is a large market and will bring large traffic to your site, but the premise is that your site is accessible to Chinese people. For indie webmasters or bloggers outside of China, that's a big challenge, because most web hosting or similar services are blocked in China, such as WordPress, Blogger, Ghost, Google App Engine, Heroku, Cloudflare and so on. If your site is hosted on a service that is accessible globally including China, you should be thankful and stick to it. If you want to test accessibility of your site in China, try [this](https://www.comparitech.com/privacy-security-tools/blockedinchina/) or [this](ce.cloud.360.cn)

# Result

This figure tells me my site now is resolved to different address inside and outside of China, don't care the red crosses status, they doesn't mean my site is blocked in China.

![My site is resolved to different address inside and outside of China](https://eastonlee.b0.upaiyun.com/blog/2017-06-30-enable-https-for-github-pages-with-custom-domain-and-stay-unblocked-in-china/Screen%20Shot%202017-06-30%20at%2010.32.11%20AM.png)

This figure tells which provinces can reach my site, green means OK, red means no-go, grey indicates there is no probe server there in that region.

![which provinces can reach my site](https://eastonlee.b0.upaiyun.com/blog/2017-06-30-enable-https-for-github-pages-with-custom-domain-and-stay-unblocked-in-china/Screen%20Shot%202017-06-30%20at%2011.55.40%20AM.png)

# I didn't tell you I've created two exactly the same content GitHub Pages

Why I created a mirror GitHub Pages for the original one? Since I've enable CNAME for the original one, 
1. if I make Cloudfront fetch page from http://eastonlee.com, then that will create an infinite CDN loop, bad things will happen, 
2. if make CF fetch from http(s)://eastonlee.github.io, GitHub server will respond a 301 redirect, and Cloudfront will fetch it, then CF returns the 301 to your browser, the redirected location and the previous location are both https://eastonlee.com , and your browser will fall into a "Too many redirect" error.

So I have to create a mirror repository and push commits to two remote repositories every time. And the mirror GitHub Pages is https://AgileSquad.github.io if you wonder, an organization page.

Why did I create an organization page, not a project page? Because if I used a project page, the URL assigned to it will be like https://eastonlee.github.io/AgileSquad, Jekyll will generate pages prepending `AgileSquad/` to many embedded URLs, causing big trouble. And luckily, the mirror organization page's CNAME file is the same and conflicts with my original one, so it will not affect the assigned page URL https://AgileSquad.github.io . For more details about GitHub Pages and custom domain, see also [here](https://help.github.com/articles/custom-domain-redirects-for-github-pages-sites/) and [here](https://help.github.com/articles/setting-up-an-apex-domain/).

# Internet Censorship in China

If you want to use a Chinese CDN provider, then you will find it very hard to find one who will be able to help without asking you to register your domain and personal information to China government's system (备案). If you'd like to register, then good luck and hope you'll never get called by some officer from **The Relevant Departments (有关部门)**. If you want to avoid registering, almost no way.

# Troubleshooting the annoying "Too Many Redirect" error when setting Cloudfront

Sometimes if the redirect keeps happening (maybe the page/DNS cache somewhere in the middle is wrong, maybe Cloudfront is not consistent globally) or only happens in certain browsers (only curl, TorBrowser, iPhone Safari get it right, no redirect, Mac Safari, Chrome, Chromium, Firefox are all wrong), try clear your browser data or reset your browser, if it's still not working, **Don't Hesitate to Use Cloudfront Invalidations!**
---
title: Bitcoin Trader bot, Part 1
layout: post
category: Machine Learning
---

I have an idea recently: predict Bitcoin's future price, and use that predication to assist my trading decision, and make profit from it. To tell if this is profitable and how profitable in academic term is hard for me, but do it is not a big deal for me. Let's just build one Bitcoin Trader bot.

If the trader bot is running, it must have some input to output some trading decision. First, it must know the historical price of Bitcoin and current price. Second, though some researchers hold the [efficient-market hypothesis](https://en.wikipedia.org/wiki/Efficient-market_hypothesis), but I don't think this hypothesis can apply on Bitcoin, so the sentiment of the crowd should not be ignored, so take account crowd's sentiment and sense it wisely.

Basic Python, R, Machine Learning knowledge is required. Because Python is such powerful general-purpose language and works like charm in scientific computing. R is born to finish statistic task and can generate beautiful charts easily. If you are not very good at math stuff and financial analysis in stock market, or not familiar stock market terms, or can't read stock index, or can't write a profitable trading strategy, I suggest you to skip the struggling for the moment, Machine Learning (aka ML )can help you a bit in this aspect, if you feed plenty amount of properly picked stock-related data to you right ML model, you can likely harvest a experienced stock expert hopefully.

Bitcoin history is not hard to get. You can get them from like quandl.com, bitcoincharts.com, data footage varies from different columns and different time granularity.

Twitter should be easy if you want to get worldwide tweets about #Bitcoin/#btc, so is stocktwits.com. When you begin to crawl data from Twitter, you should notice many limits are there, like search time range limit, API rate limit, etc. But the detail work-around is out of topic today. Sentiment but the tweets themselves is what we care, because only sentiment is measurable and computable. You can implement your own sentiment scorer but I choose existing ones. Here is a list of sentiment API provider you can choose from, and I prefer Azure because you can get sentiment scores of thousand sentences in bulks thus saving your fee: 
alchemyapi.com
sentiment.vivekn.com
[GOOGLE CLOUD NATURAL LANGUAGE API](https://cloud.google.com/natural-language/)
[HavenOnDemand](https://dev.havenondemand.com/apis)
[Azure Cognitive Services](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/text-analytics/)
Of course, that's better if you implement your sentiment scorer specific for stock/Bitcoin news, and please let me know :)

Don't be naive to directly make use of the data you just get. I suppose you have store the raw tweets/news into db, as well as their sentiment scores, but this data is a mix of noise, unrelated things and real valuable treasure. Now let's dig and wash them.
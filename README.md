# ForexTrader
Uses pattern matching to predict the future Forex market. The project is coded in goLang.
My motivation for this project is as follows. The idea of a human using technical analysis alone to trade seems silly to me for a few reasons.
1. In my view, price should be an indicator of what will happen in the future, and not what happened in the past
2. Humans are pretty good at pattern recognition, but for unnatural patterns, machines tend to be better. So it should stand that 
computers dominate technical analysis.
In this project, I developed a pattern recognition algorithm to try to predict future prices in the forex market based on previous prices. 

A brief overview of how this works:

I read in tick data (time, bid, ask, volume) and convert it to candle data. I slice this data up and store it in memory. Then, I consider an unseen
candle-time-series, find a set of similar candle-time-series and their corresponding outcomes in memory and use the outcomes to predict the future price
candle-time-series. 


A brief overview of results:

Without spread in the market, this method correctly predicts the direction of price about 60 percent of the time. However, it cannot accurately 
predict price movements larger than the spread. As such, this method predicts price movement well, but cannot be used to make profitable trades.

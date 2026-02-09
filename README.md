# 46755-Renewable_in_Electricity_Markets
""" 
Case study chosen: IEEE 24-bus reliability test system 
Details about the transmission system are available at:
https://drive.google.com/file/d/1IkRQVYhBhX1B1tP1nWjFvh4MlFc6WSkC/view?pli=1

Additional assumptions:
• Assume that the price bids of all producers are non-negative and equal to their marginal
production cost. In particular, the production cost of renewable units is assumed to be
zero. Additionally, these units offer their forecasted capacity, meaning their offer quantities
vary over time.
• For the bid price of price-elastic demands, use comparatively high values (relative to the
generation cost of conventional units) to ensure that most demands are supplied. For
inspiration, check the real bid price data in Nord Pool [link].
• A potential source for wind power forecast data is available at this link (you may normalize the data to fit your case study). Another potential source for the renewable power
generation data is renewables.ninja.
• For transmission lines, you may assume a uniform reactance for all lines (e.g., 0.002 p.u.,
leading to a susceptance of 500 p.u.)
"""

# fantasy_football_scraping
This repo has a collection of scripts made to assemble data for fantasy football analysis as well as selenium scripts to pull data from my leagues to keep track of performance of fantasy football managers

# database_assembly.ipynb
Pulls from profootball focus data of each player for specified years using beautiful soup.

Can use this to run regressions and machine learning analysis
    * Points per game next year
    * Games next year
    * Change in points per game next year
    * Change in games next year

# To do:
Does not require new scraping:
Need a targets available, targets taken measure for WR's and TE's
	Base off attempts by QB of team, tally up targets of each WR on the team last year
Need a RuAtt available, look at number of RuAtt on the team last year vs next year
Make RF into a function
Make plot of residuals
Put some cleaning back into the functions
Need a better validation function to measure performance
Can take out the least relevant variables

Requires new scraping:
* Add actual injury info, dummy variables for different injuries in the previous year(s)
* Need some sort of analysis for rookies
    * Does the team they were drafted by have a star at the same position?
    * Where were they picked in the draft?
    * Winning pct of team picked?
    * Typical college metrics including injury history
* Need some sort of comparison with ADP eventually
    * Look at ADP this year, ADP following year type of thing. Delta ADP for risers, sinkers
    * Look at second half performance and how it affects ADP??
* New coach flag

Do-able with current data:
* Can get a flag for backup RB
* "On a good team flag," maybe it's the winning percentage of the team, or a flag on > .500 winning pct



# login_pull.ipynb
Pulls from ESPN, MFL, and CBS game and season data for each fantasy manager to later be used in visualizations and analysis

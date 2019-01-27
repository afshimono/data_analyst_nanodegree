#Summary

The main idea of this dataset is to show what were the main improvemments on the credit system after
the 2008 crisis which was caused mainly by poor risk management on real state mortgage financing. This
led to high risk profiles (people with high probability of defaulting payments) being registered as low
risk, and since those securities have been considered to be low risk for a long period of time, because
it was composed of people who wanted to own their own house and not for real state speculation at first,
the market failed to realize what was happening until it was too late.

The data was taken from the suggestion of Udacity´s page 
https://docs.google.com/document/d/1w7KhqotVi5eoKE3I_AZHbsxdr-NmcWsLTIiZrpxWx4w

It comes from a financial company called prosper ( https://www.prosper.com/).

Looking at the data description on the link provided (Prosper Loan Data - Variable Definition), one can
notice some variables suffered major changes since 2009, like the Estimate Loss and Credit Grade, and 
this confirms the hypothesis of such a profound change that happened to american market, at least theoretically.

According to the overview provided on the download page, this data set contains 113,937 loans with 81 variables 
on each loan, including loan amount, borrower rate (or interest rate), current loan status, borrower income, 
and many others. 

After sending it for the first time, the reviewer suggested some changes had to be made to improve performance.
One solution found was to manually manipulate the CSV file in and extract only the data required for the 
graphs. This has made the data file smaller, from around 85 megabytes to around 282KB, and has increased performance
greatly. I am keeping both both files on the data folder though, only as a backup.

#Design

The design adopted was a Viewer-driven structure, with a simple menu on the left which contains all dimensions
available for display on a month-to-month axis. We visualize only one dimension: the ammount lended. There are 
two graph-styles: the total ammount in US dolars and the percentage in each category grouping. After looking 
through the list of available options, I chose four categories to analyse the data: income of borrowers, 
if the income is verifiable or not, if the borrower is a home owner or not, and the current status of the 
loan. Those dimensions will be enough to provide a general idea to compare the company health, specially because
many features are new and only available since 2009, which would be pointless for a pre-crisis comparison.

The main idea is that the average viewer notices that for the target institution:
- overall, loans to home owners have increased after 2008.
- there are more details about each borrower since 2007, and it seems this helped diferentiate borrowers with
lower risk of default.
- the ammount of credit currently lended is many times higher than before the crisis.


#Feedback

I showed the graph for three familiy members because I wanted to measure other aspects besides what they would
like to say about the presentation. I showed it to my wife, my mother-in-law and my brother-in-law. First, I
would like to point out that being Brazilians, only my wife is well aware of the 2008-2009 crisis in USA. For the
other 2 it was not a very significant event since one is a house-wife and the other a landlord. All of them use the 
internet, Facebook and Youtube regularly and are used to the interface of a web browser.

First impression was that it was not cleaar to all of them that the categories could be clicked and selected. After
a while observing and waiting for their comments I had to warn that they could select other categories.

Second impression was about the overall performance of dimple.js: it is extremely slow for this ammount of data.
This was important because the fact that it took a long time to switch between graphs made most of them start clicking
in other parts and I had to tell them to wait. This is a major issue with user interface and user experience 
(response time) and unfortunately it seems dimple is not optimized for large data sizes. It is important to disclose
that those observations were made before the optimization of performance, so it would take easily around 80 seconds
to load each new visualization.

One thing I missed was to make it clear what we were measuring is the ammount loaned in USD, so many of them
did not notice that and found that part a bit confusing.

After that part was cleared, they all succeded in noticing the ammount of loaned credit is bigger after 2008
than before. Only my wife made an observation about the number of details available after 2008 being larger
probably because of the crisis.


#Resources

https://www.w3schools.com/
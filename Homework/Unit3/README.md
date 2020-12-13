### HW III Project Descriptions

#### Option 1:  Boston Housing Dataset (`housing.csv`)

**Difficulty Level:** Easy
**Target Column:** PRICE

**Description:** This a numeric only dataset that contains the sale prices of 506 different houses that were sold in the Boston area.  It is the least demanding of the datasets provided,
but provides an interesting sociological backdrop to think about what drives the price of a house.  Columns are roughly separated into features that describe the house, and features that describe the
neighborhood.

**Questions that could be answered by this dataset**:  how do characteristics like proximity to work and neighborhood socioeconomic status impact housing price relative to one another?  When accounting for socioeconomic status, do things like racial or ethnic makeup have much of a remaining impact?  

**Challenges that come with this dataset:** This dataset has a large cluster of noisy observations that skew results.  Hence, it's a good opportunity to use model validation to assess the range of outcomes you might see in your out-of-sample data.  Comparisons between a traditional train-val-test set vs. KFold can illuminate the differences model validation can have on how you could interpret future performance.

#### Option 2: Insurance Premiums Dataset (`insurance_premiums.csv`)

**Difficulty Level:** Easy
**Target Columns:** Charges

**Description:** This dataset is a list of ~1500 insurance customers, their personal characteristics, and the total amount of insurance premiums they paid throughout their lifetime.  It is a mix of numeric + categorical columns, and is meant to be a lightweight introduction to customer value modeling.  The key challenges to this dataset are not just getting an accurate model score, but understanding how different customer characteristics interact with one another to drive some sort of business risk.

**Questions that could be answered by this dataset:** How do risk factors like your smoking status and BMI impact your insurance charges?  How do these types of characteristics **interact** with one another.  Ie, if you hold one constant, what is the relative impact of the other?  

**Challenges that come with this dataset:** This dataset is very balanced, in the sense that it doesn't have any truly remarkable characteristics, but is fairly complete in the way it allows you to work through the modeling process.  A good choice for someone who wants a well-rounded problem that will allow you to touch on most of the major aspects of predictive modeling without being excessive in any way.

#### Option 3:  Iowa Mini (`iowa_mini.csv`)

**Difficulty Level:** Intermediate
**Target Columns:** SalePrice

**Description**: This dataset is a somewhat amped up version of `housing.csv`, you are predicting the sale price of a house, but with a wider variety of features that go into more detail about the house itself.  It's a collection of ~1500 houses that were sold in Ames Iowa in the years 2008-2009, and the dataset is a collection of characteristics noted by the county property inspector before it was appraised.  Unlike the boston housing dataset, it comes with a mix of numeric + categorical columns, as well as a range of missing values, making it a more challenging exercise.  It's the `Iowa Mini` dataset because it only contains around 17 columns, vs. the full 88 for the entire dataset.

**Questions that could be answered by this dataset:** This dataset only includes characteristics about a particular house, so the insights will be around exactly what characteristics of a house add value.  Does having a fireplace really improve the value of a house?  What about having a build-in vs. standalone vs. carport for a garage?  

**Challenges that come with this dataset:** This dataset provides lots of challenges for data encoding.  It comes with a variety of numeric + categorical data, and you have to decide how to properly encode them......including their missing values.  It is a good exercise in making distinctions about random vs. non-random null values, and how choosing the appropriate option can impact your results.  Like the Bostong Housing dataset, it also contains outliers, so it's a useful exercise to see how they impact your results and how model validation can allow you to estimate out-of-sample performance.

**Outside sources for this dataset:** This dataset is part of an ongoing Kaggle Competition, which is listed here:  https://www.kaggle.com/c/house-prices-advanced-regression-techniques

#### Option 4:  Bikeshare (`bikeshare.csv`)

**Difficulty Level:** Intermediate
**Target Column:** count

**Description:** This dataset counts hourly bike rentals throughout a city over the course of 2 years.  It contains around 10,000 samples that logs information about the weather at that point in time, as well as the date.  It's meant to be a lightweight time series that gives people practice in encoding time based variables, but without the messiness of the `restaurants` data set.

**Questions that could be answered by this dataset:** How do time based characteristics like time of day and previously occurring trends impact the ability to forecast total rentals?  What about weather?  This dataset is primarily about being able to accurately capture the impact of time on your target variable.  A good choice for people who want to continue on the forecasting path, but want a change of pace from the data covered in class so far.

**Challenges that come with this dataset:** Success on this dataset heavily relies on capturing the passage of time.  Doing this correctly will give you good results, if you miss the key points your model will punch well below its weight.  The dataset is also setup in a slightly unusual way:  it only includes days 1-18 of the month, so if you are using shift statistics it's important to make note of this.

**Outside sources for this dataset:** This dataset is listed on kaggle, and has many active notebooks, which you can find here:  https://www.kaggle.com/c/bike-sharing-demand

#### Option 5:  Iowa Full (`iowa_full.csv`)

**Difficulty Level:** Difficult
**Target Column:** SalePrice

**Description:** This dataset is the full version of the dataset used in  `iowa_mini.csv`.  Its meaning is the same, except the number of columns used has been greatly expanded to include all the information included in the property appraisal.

**Questions that could be answered by this dataset:** See above.

**Challenges that come with this dataset:** See above, with the exception that they are much greater.  This dataset is a painful exercise in working through messy, erratic data and encoding it properly.

**Outside sources for this dataset:** See above.

#### Option 6:  Restaurants(`restaurants.csv`)

**Difficulty Level:** Difficult
**Target Column:** visitors

**Description:** This is the dataset we've been using throughout class.  It should be familiar by now, but if we need more context:  this dataset is a collection of 829 different restaurants, and their attendance over 2 year timespan.  Included data is information about the restaurant's genre, location (both categorical & latitude + longitude coordinates), as well as reservations for that day and a date column.

**Questions that could be answered by this dataset:** Attendance or similar measures (like customers or sales) are common business objectives that are trying to be optimized, so building a useful model that captures what's driving it is a very applicable business use case.  What are the different characteristics of time and its corresponding trends that better allow us to understand what we should expect to happen with attendance?

**Challenges with this dataset:** This is the only dataset that is both a time series, as well as hierarchical.  The hierarchical nature of the dataset makes it especially challenging, because steps you'd have to take in order to add features or process it will be more complicated and will require techniques learned in previous labs.  It's also the largest dataset, so additional precautions will need to be taken in order to fit your model and do an effective parameter search.

**Outside sources for this dataset:** This dataset is an actively studied kaggle dataset, which can be found here:  https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting

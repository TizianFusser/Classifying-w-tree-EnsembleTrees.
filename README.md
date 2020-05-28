# Classifying-w-tree-EnsembleTrees.
Testing the performance of RandomForest, ADAboosting and gradientBoosting on a classification task.


Business Problem 
The dataset contains values from a Website that shows the usage of the webpages. The different webpages are categorized. There is a binary variable 'transaction'; this variable shows if there is a transaction at the end of the session or not. The transaction is going to be our target variable. We analyze the data to recommend increasing the transactions per session.

Data preparation

Converting the data from categorical to numerical values. Next step, checking for correlations: there are a bunch of highly correlated variables, for every category, there is one feature how many pages of each category was visited and how much time was spend ion this category. I decided to keep the time values because they give a better insight into how much information was gained from the user in each category. The Exit Rate and the Bounce Rate are highly correlated too, and we are dropping the Bounce Rate because there is more information in the Exit Rate.

The dataset is going to be normalized and balanced. Afterward, we split the data into a train80/test20 ratio.

Model Evaluation

To tackle the problem with the low percentage of transactions, only 16% of the sessions have a transaction. We can't use the accuracy score for evaluating the model. We are decided to reduce the FalseNegative if we have a high FN value. We would lose money because we are predicting people are not doing a transaction, but actually, they would do. In addition, we are able to evaluate the model by the AUC score, which can be used in imbalanced datasets
We are running three different decision tree models, RandomForest, wich bases on the concept of bagging. ADAboosting and GradientBoosting, which is based on the idea of boosting.

The random forest performs the worst with an FN 112 and an AUC score of 0.795.

ADAboosting gives us the best results: FN 85 and AUC score of 0.826

GradientBoosting creates a: FN 99 and an AUC score of 0.813

 We are comparing the significances of the features in the different models:
RandomForest			         ADAboosting			      GradientBoosting





The ADAboosting Modle is going to be deployed and used for the recommendations; however, the GradienBoosting is taken into consideration because the values are very similar, and as shown in the figures, the feature significance is different.  The most significant values in ADAboost are Month, Exit Rate, and Page Value. In the gradient boost model, PageValue is most significant. Random Forest performed the worst because it always grows a complete tree, features which are not substantial are taken into consideration too much with this process. ADAboosting gives us very smooth values because it only takes one stump in consideration for the significance of each feature. The sequence calculates the weightage of each feature, and the less significant features don't mess up our FN and AUC score. The best performing is the GradientBoost algorithm because it is tuned by grid-search, and we don't need to grow complete trees and cut the leaves down.


Recommendation

As we can see in the figure the most of the transactions were made before Christmas. In order to increase the transaction, it should be taken into consideration to do special promotion over the summer months.
The concept of the Page Value belongs to how much value can give a single page to the whole shopping basket. Pages leading to a transaction have a higher Page Value like the shopping basket has a high Page Value. The recommendation is to make a dashboard on the landing page with links or ads of pages with products or information with a top page value. A transaction can be increased with this mechanic. 

The boxplot shows us longer ProductRelated_Duration increases the chance a transaction is made, to increase the sales we can promote product related pages on social media and install the links on the landing page. 
 
Administrative Duration plays a role in the transaction mechanic. Still, instead, to keep customers on the administrative page, I would make the information as easily accessible as possible in a sidebar to keep the shopping spree flowing.

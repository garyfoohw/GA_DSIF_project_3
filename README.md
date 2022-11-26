# ![](https://awaypoint.files.wordpress.com/2015/08/christian-vs-atheist.jpg)

# Keeping Religions Safe

### Background

Differences in religious belief has always been the sensitive topic, no matter when in history or where in the world. Differences in beliefs by itself is not a problem. However, when people try to impose their own beliefs on others, then that is a recipe for disharmony or something worse.

It is therefore important to protect harmony and peace by identifying individuals who may be of a different religion from another, or if we generalize the problem, one with religion vs one without. The ability to identify one's religion (or lackthereof) allow organizers to identify who to invite for events and who to keep out.

In this manner, we can be sure that each individual groups can carry out their activities without interference or trouble.

### Problem Statement

A machine learning algorithm will be trained to identify natural speech of _Christians_ (as a proxy for people with religion) vs _Aetheists_.

By studying the common topic / words of choice between these 2 groups of people, it is hoped that a model can identify one from the other.

### The process

Posts from Reddit `christians` and `aethiest` are scraped.
The posts are subjected to some cleaning and engineering, including removal of URLs and NLP lemmatization, and use of english stopwords.
The posts are vectorised using both a normal `CounterVectorizer` and `TfidfVectorizer`.
Finally, we pass the model through several classification models.

### Results

| Model         | Vectorizer      | ROC AUC score |
| ------------- | --------------- | ------------- |
| Naive Bayes   | CountVectorizer | 81.2%         |
| Naive Bayes   | TfidfVectorizer | 82.6%         |
| Random Forest | TfidfVectorizer | 83.5%         |
| XGBoost       | TfidfVectorizer | 83.1%         |

All 3 models learn rather well with Random Forest and XGBoost scoring rather closely.
We choose XGBoost as the final model despite the slightly lower score due to its faster runtime performance.

### Recommendation and Conclusion

It is recommended that this model be further tuned, since decision trees have many parameters that are tunable.
Possible to also try other classifiers including Neural Network.
Finally, the confusion matrix was not studied in-depth in the interest of time, and it would be prudent to study the precision, recall, and specificity too.

In conclusion, we have built a model to learn natural text used by aethiests and christians through posts on Reddit. The ROC score achieved at least 83% using a XGBoost Classifier.

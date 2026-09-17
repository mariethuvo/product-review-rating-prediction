# Product Review Rating Prediction
 
Multi-class classification project predicting a customer's 1–5 star rating from draft review text and product metadata, built as a case study on ShopSphere, a large e-commerce marketplace. The goal was to flag likely dissatisfied customers (1–2 star reviews) before they post, enabling proactive outreach instead of after-the-fact damage control.
 
## Business Problem
 
Customer reviews drive purchase decisions and search rankings on the platform, but by the time a bad review is posted, the damage is done. The task: predict the star rating a customer is about to leave, using their draft review text plus structured metadata (price, seller rating, delivery time, etc.), so the Customer Experience and Seller Success teams can intervene early — particularly for the 1- and 2-star cases, where a missed intervention means a lost chance to resolve the issue before it becomes public.
 
This is a multi-class classification problem with an imbalanced target: ratings 1 and 2 make up only 10.85% and 14.60% of the data, respectively, while the majority of reviews cluster at 3–4 stars.
 
## Data
 
2,000 labeled reviews with review text plus structured features (product price, category, seller rating, delivery days, product age, reviewer's prior review count, verified purchase flag).
 
## Approach
 
**Text EDA before modeling.** Vocabulary size dropped from 411 to 324 words after basic cleaning (lowercasing, punctuation removal), meaning a small, high-signal vocabulary, which shaped the TF-IDF configuration below.
 
**Feature pipeline**, built to avoid data leakage at every step:
- TF-IDF vectorizer fit on training data only, `max_features=5000`, `ngram_range=(1,2)` to capture negation phrases like "not good," `min_df=5` to drop noise terms
- TruncatedSVD applied after TF-IDF to reduce dimensionality
- Structured features scaled with `StandardScaler` inside the same pipeline, so scaling is refit per cross-validation fold rather than leaking test-fold statistics into training
**Ablation study** — the key methodological piece of this project. Before picking a model, I tested three feature configurations (structured-only, text-only, combined) across every model type, to check whether combining text and structured data was actually worth the added complexity:
 
| Model | Structured-only | Text-only | Combined | Δ vs. best single modality |
|---|---|---|---|---|
| Logistic Regression | 0.512 | 0.687 | 0.722 | +0.034 |
| Random Forest | 0.489 | 0.631 | 0.668 | +0.037 |
| XGBoost | 0.534 | 0.698 | 0.739 | +0.041 |
 
Text alone dominates — structured features add a real but modest lift (+0.03 to +0.04 weighted F1) on top of text. That's a useful finding on its own: it means the metadata is worth keeping, but review text is doing most of the work.
 
**Model exploration and tuning.** Compared Logistic Regression, Random Forest, and a Voting Classifier ensemble on the combined feature set; the Voting Classifier came out on top with a weighted F1 of 0.60. I then jointly tuned a HistGradientBoostingClassifier's hyperparameters *together with* the TF-IDF and SVD parameters (via `RandomizedSearchCV`), rather than tuning the model in isolation — this pushed weighted F1 to 0.6068, a small but real gain over the ensemble.
 
**Final model:** the tuned HistGradientBoostingClassifier, chosen over the marginally-tied Voting Classifier because it's a single streamlined pipeline rather than three models running in parallel, and because it could be tuned jointly with the text representation, which the Voting Classifier couldn't.
 
**Threshold calibration.** Since missing a dissatisfied customer (rating 1–2) is more costly than a false alarm, I tested custom decision thresholds based on the combined predicted probability of ratings 1 and 2, rather than defaulting to argmax — trading some precision for better recall on the classes that actually matter to the business.
 
## Results
 
- **Final weighted F1: 0.6068** (tuned HistGradientBoosting, combined features)
- Ablation study confirmed combined features beat single-modality on every model tested
## Error Analysis
 
- The model struggles most on "deceptive" reviews (~10% of data) where the text sentiment contradicts the assigned rating — a case where the label itself is arguably noisy, not just hard to predict
- Adjacent ratings (1 vs. 2, 4 vs. 5) are the most common confusions
- Ratings 1 and 2 are harder to predict due to lower support in training data; the model also tends to play it safe and predict 4 instead of 5 on lukewarm-positive text
- Short reviews (under 45 words) provide less signal for the text pipeline to work with
## What I'd do differently
 
The "deceptive review" failure mode suggests the label quality itself is worth investigating before pushing the model further — no amount of tuning fixes a case where text and rating genuinely disagree. I'd also want more training data specifically for the 1–2 star classes, since their low support is likely a bigger constraint on recall than model choice at this point.
 
## Tech Stack
 
Python, Pandas, NumPy, Scikit-learn (TF-IDF, TruncatedSVD, HistGradientBoostingClassifier, VotingClassifier), Matplotlib, Seaborn
 
## Files
 
- `VoMarie_ProductReviewRatingPrediction.ipynb` 

<html>
<head>
<title>Earthquake Predictor Neural Network</title>
</head>
<body>
<?php include("header.php"); ?>
<?php include("nav.php"); ?>
<h2>Neural Network Model Information</h2>
<p>(This is an archive of the first version of the model utilized in 2018.)</p>
<h3>Dataset:</h3>
<p>
Source: United States Geological Survey (USGS)<br>
Training Set: 1973-01-01 to 2016-12-31<br>
Test Set: 2017-01-01 to 2017-12-31<br>
Update Set: 2018-01-01 to 2018-02-28<br>
Seed Set: Last 1 Day (Previous 24 Hours)<br>
Temporal Resolution: 1 Day (24 Hours)<br>
Spatial Resolution: 2 Degrees Latitude and 2 Degrees Longitude<br>
Details: Earthquakes were reorganized into tensors, with each slice representing a single day and consisting of a matrix with the rows and columns representing approximate latitudes and longitudes.  Each position in the matrix was filled with a magnitude determined by converting all the earthquakes occurring at that approximate location and time from logarithmic scale magnitudes into linear energies, summing them together, and then converting the resulting value back into the logarithmic scale.<br>
</p>
<h3>Architecture:</h3>
<p>
Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)<br>
Hidden Layers: 5<br>
Neurons Per Hidden Layer: 512 nodes<br>
Timesteps: 1<br>
Epochs of Training: 1<br>
Loss Function: Asymmetric (Exponential Or Logarithmic)<br>
Library: Keras with Theano backend<br>
Notes: Model is fully Stateful and utilizes an Online Learning schedule. Utilizes Layer Normalization and Residual Skip Connections as well as Golden Ratio scaling of the activation functions, and Golden Ratio Conjugate scaling of the gradient norms. No dropout or stochastic timeskip.
</p>
<h3>Statistics:</h3>
<h4>Prediction Magnitudes:</h4>
<ul>
<li>0.0-0.5: 3075100</li>
<li>0.5-1.0: 22426</li>
<li>1.0-1.5: 15027</li>
<li>1.5-2.0: 14846</li>
<li>2.0-2.5: 13726</li>
<li>2.5-3.0: 8186</li>
<li>3.0-3.5: 7389</li>
<li>3.5-4.0: 5913</li>
<li>4.0-4.5: 7375</li>
<li>4.5-5.0: 367</li>
<li>5.0-5.5: 1</li>
<li>5.5-6.0: 0</li>
<li>6.0-6.5: 0</li>
<li>6.5-7.0: 0</li>
<li>7.0-7.5: 0</li>
<li>7.5-8.0: 0</li>
<li>8.0-8.5: 0</li>
<li>8.5-9.0: 0</li>
<li>9.0-9.5: 0</li>
</ul>
<ul>
<li>A True Positive is scored when the magnitude prediction is within +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is greater than 0.0.</li>
<li>A True Negative is scored when the magnitude prediction is within +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is less than 0.0.</li>
<li>A False Positive is scored when the magnitude prediction is outside of +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is less than the predicted magnitude.</li>
<li>A False Negative is scored when the magnitude prediction is outside of +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is greater than the predicted magnitude.</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 0.0 Or Greater</h4>
<ul>
<li>True Positive (TP) = 21134.0</li>
<li>True Negative (TN) = 5912431.0</li>
<li>False Positive (FP) = 49833.0</li>
<li>False Negative (FN) = 12046.0</li>
<li>Total Population (TP + TN + FP + FN) = 5995444.0</li>
<li>Positive (TP + FN) = 33180.0</li>
<li>Negative (FP + TN) = 5962264.0</li>
<li>Predict Positive (TP) = 70967.0</li>
<li>Predict Negative (TP) = 5924477.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.00553420230428</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.636949969861</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.363050030139</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 0.00835806666729</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.991641933333</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.989678996251</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.297800386095</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.702199613905</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 0.00203325964469</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.997966740355</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 76.2078115928</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0.366110001942</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 208.155502959</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.25458663101</li>
<li>F1-Score = 0.405849424371</li>
<li>Matthews Correlation Coefficient = 0.431180728834</li>
<li>Informedness = 0.628591903194</li>
<li>Markedness = 0.29576712645</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 5.0 Or Greater</h4>
(Note: Some statistics cannot be properly assigned because of the lack of true negatives. Also, these numbers are somewhat biased due to the sparsity of predictions above 5.0)
<ul>
<li>True Positive (TP) = 95.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 0.0</li>
<li>False Negative (FN) = 1279.0</li>
<li>Total Population (TP + TN + FP + FN) = 1374.0</li>
<li>Positive (TP + FN) = 1374.0</li>
<li>Negative (FP + TN) = 0.0</li>
<li>Predict Positive (TP) = 95.0</li>
<li>Predict Negative (TP) = 1279.0</li>
<li>Prevalence (PRE = positive / total_pop) = 1.0</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.0691411935953</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.930858806405</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.0691411935953</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 1.0</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.0</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.0691411935953</li>
<li>F1-Score = 0.129339686862</li>
<li>Matthews Correlation Coefficient = 0</li>
<li>Informedness = -0.930858806405</li>
<li>Markedness = 0.0</li>
</ul>
<?php include("footer.php"); ?>
</body>
</html>
<h2>High-Recall Model</h2>
<h3>Architecture:</h3>
<p>
Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)<br>
Hidden Layers: 5<br>
Neurons Per Hidden Layer: 512 nodes<br>
Timesteps: 1<br>
Epochs of Training: 1<br>
Loss Function: Asymmetric (Exponential Or Logarithmic)<br>
Library: Keras with Theano backend<br>
Notes: Model is fully Stateful and utilizes an Online Learning schedule. Utilizes a custom architecture that combines dense connectivity with residual skip connections, as well as a scaling the norm of the gradients.
</p>
<h3>Statistics:</h3>
<h4>Prediction Magnitudes:</h4>
<ul>
<li>0.0-0.5: 10772</li>
<li>0.5-1.0: 9674</li>
<li>1.0-1.5: 10365</li>
<li>1.5-2.0: 12663</li>
<li>2.0-2.5: 18525</li>
<li>2.5-3.0: 20084</li>
<li>3.0-3.5: 17291</li>
<li>3.5-4.0: 12215</li>
<li>4.0-4.5: 11393</li>
<li>4.5-5.0: 6279</li>
<li>5.0-5.5: 516</li>
<li>5.5-6.0: 48</li>
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
(Note: Some statistics cannot be properly assigned because of the lack of true negatives.)
<ul>
<li>True Positive (TP) = 24352.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 100784.0</li>
<li>False Negative (FN) = 10555.0</li>
<li>Total Population (TP + TN + FP + FN) = 135691.0</li>
<li>Positive (TP + FN) = 34907.0</li>
<li>Negative (FP + TN) = 100784.0</li>
<li>Predict Positive (TP) = 125136.0</li>
<li>Predict Negative (TP) = 10555.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.257253612988</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.697625118171</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.302374881829</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 1.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.179466582161</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.194604270554</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.805395729446</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0.697625118171</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.179466582161</li>
<li>F1-Score = 0.30431821448</li>
<li>Matthews Correlation Coefficient = 0.0</li>
<li>Informedness = -0.302374881829</li>
<li>Markedness = -0.805395729446</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 5.0 Or Greater</h4>
(Note: Some statistics cannot be properly assigned because of the lack of true negatives. Also, these numbers are somewhat biased due to the sparsity of predictions above 5.0)
<ul>
<li>True Positive (TP) = 499.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 333.0</li>
<li>False Negative (FN) = 1673.0</li>
<li>Total Population (TP + TN + FP + FN) = 2505.0</li>
<li>Positive (TP + FN) = 2172.0</li>
<li>Negative (FP + TN) = 333.0</li>
<li>Predict Positive (TP) = 832.0</li>
<li>Predict Negative (TP) = 1673.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.867065868263</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.229742173112</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.770257826888</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 1.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.199201596806</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.599759615385</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.400240384615</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0.229742173112</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.199201596806</li>
<li>F1-Score = 0.332223701731</li>
<li>Matthews Correlation Coefficient = 0</li>
<li>Informedness = -0.770257826888</li>
<li>Markedness = -0.400240384615</li>
</ul>
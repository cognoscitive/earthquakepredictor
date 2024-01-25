<h2>High-Precision Model</h2>
<h3>Architecture:</h3>
<p>
Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)<br>
Hidden Layers: 5<br>
Neurons Per Hidden Layer: 512 nodes<br>
Timesteps: 1<br>
Epochs of Training: 1<br>
Loss Function: Asymmetric (L2 Or Smooth L1)<br>
Library: Keras with Theano backend<br>
Notes: Model is fully Stateful and utilizes an Online Learning schedule. Utilizes a custom architecture that combines dense connectivity with residual skip connections, as well as a scaling the norm of the gradients.
</p>
<h3>Statistics:</h3>
<h4>Prediction Magnitudes:</h4>
<ul>
<li>0.0-0.5: 4804</li>
<li>0.5-1.0: 4489</li>
<li>1.0-1.5: 4979</li>
<li>1.5-2.0: 6406</li>
<li>2.0-2.5: 8116</li>
<li>2.5-3.0: 9046</li>
<li>3.0-3.5: 6975</li>
<li>3.5-4.0: 2735</li>
<li>4.0-4.5: 577</li>
<li>4.5-5.0: 152</li>
<li>5.0-5.5: 56</li>
<li>5.5-6.0: 25</li>
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
<li>True Positive (TP) = 17621.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 24710.0</li>
<li>False Negative (FN) = 17801.0</li>
<li>Total Population (TP + TN + FP + FN) = 60132.0</li>
<li>Positive (TP + FN) = 35422.0</li>
<li>Negative (FP + TN) = 24710.0</li>
<li>Predict Positive (TP) = 42331.0</li>
<li>Predict Negative (TP) = 17801.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.589070711102</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.497459206143</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.502540793857</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 1.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.293038648307</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.416267038341</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.583732961659</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0.497459206143</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.293038648307</li>
<li>F1-Score = 0.453255822926</li>
<li>Matthews Correlation Coefficient = 0.0</li>
<li>Informedness = -0.502540793857</li>
<li>Markedness = -0.583732961659</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 5.0 Or Greater</h4>
(Note: Some statistics cannot be properly assigned because of the lack of true negatives. Also, these numbers are somewhat biased due to the sparsity of predictions above 5.0)
<ul>
<li>True Positive (TP) = 118.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 0.0</li>
<li>False Negative (FN) = 1965.0</li>
<li>Total Population (TP + TN + FP + FN) = 2083.0</li>
<li>Positive (TP + FN) = 2083.0</li>
<li>Negative (FP + TN) = 0.0</li>
<li>Predict Positive (TP) = 118.0</li>
<li>Predict Negative (TP) = 1965.0</li>
<li>Prevalence (PRE = positive / total_pop) = 1.0</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.0566490638502</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.94335093615</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 0.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.0566490638502</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 1.0</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.0</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.0566490638502</li>
<li>F1-Score = 0.107223989096</li>
<li>Matthews Correlation Coefficient = 0</li>
<li>Informedness = -0.94335093615</li>
<li>Markedness = 0.0</li>
</ul>
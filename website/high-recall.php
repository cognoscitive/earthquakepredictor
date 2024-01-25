<h2>High-Recall Model</h2>
<h3>Architecture:</h3>
<p>
Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)<br>
Hidden Layers: 5<br>
Neurons Per Hidden Layer: 512 nodes<br>
Timesteps: 1<br>
Epochs of Training: 1<br>
Loss Function: Asymmetric (Exponential Or Logarithmic)<br>
Library: Keras in Tensorflow 2.1<br>
Notes: Model is fully Stateful and utilizes an Online Learning schedule. Utilizes a dual path architecture that combines dense connectivity with residual skip connections.
</p>
<h3>Statistics:</h3>
<h4>Prediction Magnitudes:</h4>
<ul>
<li>-1.0--0.5: 3204425</li>
<li>-0.5-0.0: 92150</li>
<li>0.0-0.5: 44708</li>
<li>0.5-1.0: 25858</li>
<li>1.0-1.5: 19453</li>
<li>1.5-2.0: 17694</li>
<li>2.0-2.5: 19633</li>
<li>2.5-3.0: 19833</li>
<li>3.0-3.5: 17503</li>
<li>3.5-4.0: 13259</li>
<li>4.0-4.5: 10300</li>
<li>4.5-5.0: 5650</li>
<li>5.0-5.5: 1034</li>
<li>5.5-6.0: 138</li>
<li>6.0-6.5: 4</li>
<li>6.5-7.0: 0</li>
<li>7.0-7.5: 0</li>
<li>7.5-8.0: 0</li>
<li>8.0-8.5: 0</li>
<li>8.5-9.0: 0</li>
<li>9.0-9.5: 0</li>
<li>9.5-10.0: 0</li>
</ul>
<ul>
<li>A True Positive is scored when the magnitude prediction is within +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is greater than 0.0.</li>
<li>A True Negative is scored when the magnitude prediction is within +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is less than 0.0.</li>
<li>A False Positive is scored when the magnitude prediction is outside of +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is less than the predicted magnitude.</li>
<li>A False Negative is scored when the magnitude prediction is outside of +/-1.0 of the actual earthquake event magnitude, and the actual earthquake event magnitude is greater than the predicted magnitude.</li>
</ul>
<h4>All Magnitudes</h4>
<ul>
<li>True Positive (TP) = 23415.0</li>
<li>True Negative (TN) = 3293568.0</li>
<li>False Positive (FP) = 164801.0</li>
<li>False Negative (FN) = 11082.0</li>
<li>Total Population (TP + TN + FP + FN) = 3492866.0</li>
<li>Positive (TP + FN) = 34497.0</li>
<li>Negative (FP + TN) = 3458369.0</li>
<li>Predict Positive (TP) = 188216.0</li>
<li>Predict Negative (TP) = 3304650.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.009876416673299233</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.6787546743195061</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.32124532568049396</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 0.0476528097493356</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.9523471902506644</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.949645076564632</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.12440493900624815</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.8755950609937518</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 0.0033534564931233263</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.9966465435068766</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 14.243749275014569</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0.337319550022445</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 42.2262785363813</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.11748738070627904</li>
<li>F1-Score = 0.21027061734160107</li>
<li>Matthews Correlation Coefficient = 0.2763979311120408</li>
<li>Informedness = 0.6311018645701705</li>
<li>Markedness = 0.12105148251312481</li>
<li>Balanced Accuracy = 0.8155509322850852</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 5.0 Or Greater</h4>
(Note: Some statistics cannot be properly assigned because of the lack of true negatives. These numbers are somewhat biased due to the sparsity of predictions above 5.0)
<ul>
<li>True Positive (TP) = 578.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 829.0</li>
<li>False Negative (FN) = 1678.0</li>
<li>Total Population (TP + TN + FP + FN) = 3085.0</li>
<li>Positive (TP + FN) = 2256.0</li>
<li>Negative (FP + TN) = 829.0</li>
<li>Predict Positive (TP) = 1407.0</li>
<li>Predict Negative (TP) = 1678.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.7312803889789303</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.25620567375886527</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.7437943262411347</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 1.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.1873581847649919</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.4108031272210377</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.5891968727789624</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0.25620567375886527</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.1873581847649919</li>
<li>F1-Score = 0.3155883155883156</li>
<li>Matthews Correlation Coefficient = 0</li>
<li>Informedness = -0.7437943262411347</li>
<li>Markedness = -0.5891968727789623</li>
<li>Balanced Accuracy = 0.12810283687943264</li>
</ul>
<h2>High-Precision Model</h2>
<h3>Architecture:</h3>
<p>
Long Short Term Memory (LSTM) Recurrent Neural Network (RNN)<br>
Hidden Layers: 5<br>
Neurons Per Hidden Layer: 512 nodes<br>
Timesteps: 1<br>
Epochs of Training: 1<br>
Loss Function: Symmetric (Smooth L1)<br>
Library: Keras in Tensorflow 2.1<br>
Notes: Model is fully Stateful and utilizes an Online Learning schedule. Utilizes a dual path architecture that combines dense connectivity with residual skip connections.
</p>
<h3>Statistics:</h3>
<h4>Prediction Magnitudes:</h4>
<ul>
<li>(-1.0)-(-0.5): 2938222</li>
<li>(-0.5)-0.0: 12102</li>
<li>0.0-0.5: 2795</li>
<li>0.5-1.0: 1460</li>
<li>1.0-1.5: 1513</li>
<li>1.5-2.0: 2307</li>
<li>2.0-2.5: 3906</li>
<li>2.5-3.0: 5784</li>
<li>3.0-3.5: 4563</li>
<li>3.5-4.0: 1868</li>
<li>4.0-4.5: 537</li>
<li>4.5-5.0: 236</li>
<li>5.0-5.5: 93</li>
<li>5.5-6.0: 27</li>
<li>6.0-6.5: 0</li>
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
<li>True Positive (TP) = 14602.0</li>
<li>True Negative (TN) = 2939771.0</li>
<li>False Positive (FP) = 7173.0</li>
<li>False Negative (FN) = 20777.0</li>
<li>Total Population (TP + TN + FP + FN) = 2982323.0</li>
<li>Positive (TP + FN) = 35379.0</li>
<li>Negative (FP + TN) = 2946944.0</li>
<li>Predict Positive (TP) = 21775.0</li>
<li>Predict Negative (TP) = 2960548.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.01186290016205488</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.41273071596144606</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.5872692840385539</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 0.00243404693132954</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.9975659530686705</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.9906281110396158</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.670585533869116</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.32941446613088404</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 0.007017957486249168</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.9929820425137509</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 169.565635998646</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0.5887022128532162</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 288.03295162901753</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.34315660838503476</li>
<li>F1-Score = 0.5109703607796481</li>
<li>Matthews Correlation Coefficient = 0.5217849808745723</li>
<li>Informedness = 0.4102966690301164</li>
<li>Markedness = 0.6635675763828668</li>
<li>Balanced Accuracy = 0.7051483345150582</li>
</ul>
<h4>Either Prediction or Actual Magnitude Is 5.0 Or Greater</h4>
(Note: Some statistics cannot be properly assigned because of the lack of true negatives. These numbers are somewhat biased due to the sparsity of predictions above 5.0)
<ul>
<li>True Positive (TP) = 167.0</li>
<li>True Negative (TN) = 0.0</li>
<li>False Positive (FP) = 4.0</li>
<li>False Negative (FN) = 1936.0</li>
<li>Total Population (TP + TN + FP + FN) = 2107.0</li>
<li>Positive (TP + FN) = 2103.0</li>
<li>Negative (FP + TN) = 4.0</li>
<li>Predict Positive (TP) = 171.0</li>
<li>Predict Negative (TP) = 1936.0</li>
<li>Prevalence (PRE = positive / total_pop) = 0.9981015662078785</li>
<li>True Positive Rate OR Sensitivity OR Recall (TPR = TP / positive) = 0.07941036614360437</li>
<li>False Negative Rate OR Miss Rate (FNR = FN / positive) = 0.9205896338563956</li>
<li>False Positive Rate OR Fall-Out (FPR = FP / negative) = 1.0</li>
<li>True Negative Rate OR Specificity (TNR = TN / negative) = 0.0</li>
<li>Accuracy (ACC = (TP + TN) / total_pop) = 0.07925961082107262</li>
<li>Positive Predictive Value OR Precision (PPV = TP / predict_positive) = 0.9766081871345029</li>
<li>False Discovery Rate (FDR = FP / predict_positive) = 0.023391812865497075</li>
<li>False Omission Rate (FOR = FN / predict_negative) = 1.0</li>
<li>Negative Predictive Value (NPV = TN / predict_negative) = 0.0</li>
<li>Positive Likelihood Ratio (PLR = TPR / FPR) = 0.07941036614360437</li>
<li>Negative Likelihood Ratio (NLR = FNR / TNR) = 0</li>
<li>Diagnostic Odds Ratio (DOR = PLR / NLR) = 0</li>
<li>Jaccard Index (IoU) (J = TP / (TP + FP + FN)) = 0.07925961082107262</li>
<li>F1-Score = 0.14687774846086193</li>
<li>Matthews Correlation Coefficient = 0</li>
<li>Informedness = -0.9205896338563956</li>
<li>Markedness = -0.023391812865497075</li>
<li>Balanced Accuracy = 0.03970518307180219</li>
</ul>
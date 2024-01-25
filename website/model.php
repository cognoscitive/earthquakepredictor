<html>
<head>
<title>Earthquake Predictor Neural Network</title>
</head>
<body>
<?php include("header.php"); ?>
<?php include("nav.php"); ?>
<h2>Neural Network Model Information</h2>
<p>Initially trained on: 2020-03-17</p>
<p>This is the latest version of the model. Older versions were used in <a href="model_v0.php">2018</a> and <a href="model_v1.php">2019</a>.</p>
<h3>Dataset:</h3>
<p>
Source: United States Geological Survey (USGS)<br>
Training Set: 1973-01-01 to 2017-12-31<br>
Test Set: 2018-01-01 to 2018-12-31<br>
Update Set: 2019-01-01 to 2020-03-16<br>
Seed Set: Last 1 Day (Previous 24 Hours)<br>
Temporal Resolution: 1 Day (24 Hours)<br>
Spatial Resolution: 2 Degrees Latitude and 2 Degrees Longitude<br>
Details: Earthquakes were reorganized into tensors, with each slice representing a single day and consisting of a matrix with the rows and columns representing approximate latitudes and longitudes.  Each position in the matrix was filled with a magnitude determined by converting all the earthquakes occurring at that approximate location and time from logarithmic scale magnitudes into linear energies, summing them together, and then converting the resulting value back into the logarithmic scale. Some preprocessing is required to convert between different types of magnitude scales as well. This model is designed for the moment magnitudes (Mw). A baseline magnitude of -1.0 is assumed for areas where no event takes place.<br>
</p>
<form>
Model: 
<?php
	$model = $_GET["model"];
	if ($model == '') {
		$model = 'high-recall';
	}
?>
<select name="model">
<option value="high-recall" <?php if ($model == 'high-recall') { echo 'selected'; } ?> >high-recall</option>
<option value="high-precision" <?php if ($model == 'high-precision') { echo 'selected'; } ?> >high-precision</option>
</select>
 <input type="submit" value="Submit">
</form>
<?php 
if ($model == "high-recall") {
	include("high-recall.php");
} 
if ($model == "high-precision") {
	include("high-precision.php");
} 
if ($model == "") {
	echo "<p>- PLEASE CHOOSE A MODEL -</p>";
}  
?>
<?php include("footer.php"); ?>
</body>
</html>
<?php include("footer.php"); ?>
</body>
</html>
<html>
<head
><title>Earthquake Predictor Neural Network</title>
<style>	#map {	width: 1080px;	height: 540px;	} </style>
</head>
<body>
<?php include("header.php"); ?>
<?php include("nav.php"); ?>
<p>This is an archive of previous predictions.</p>
<form>
<?php
	$model = $_GET["model"];
	if ($model == '') {
		$model = 'high-recall';
	}
?>
Model: 
<select name="model">
<option value="high-recall" <?php if ($model == 'high-recall') { echo 'selected'; } ?> >high-recall</option>
<option value="high-precision" <?php if ($model == 'high-precision') { echo 'selected'; } ?> >high-precision</option>
</select>
 Date: 
<select name="date">
<option value=""></option>
<?php
	$file = fopen($model . '/archive.txt', 'r');
	while(($option = fgets($file)) !== FALSE) {
		if (($date = $_GET["date"])) {
			if (trim($option) == $date) {
				echo '<option value="' . trim($option) . '" selected >' . trim($option) . '</option>';
			} else {
				echo '<option value="' . trim($option) . '">' . trim($option) . '</option>';
			}
		} else {
			echo '<option value="' . trim($option) . '">' . trim($option) . '</option>';
		}
	}
	fclose($file);
	$minimum = $_GET["minimum"];
	if ($minimum == "") {
		$minimum = 4.0;
	}
	$mindelta = $_GET["mindelta"];
	if ($mindelta == "") {
		$mindelta = 0.5;
	}
	$maptype = $_GET["maptype"];
	if ($maptype == '') {
		$maptype = 'magnitudes';
	}
?>
</select>
 Minimum Magnitude Cutoff: 
<select name="minimum">
<option value="0.0" <?php if ($minimum == 0.0) { echo 'selected'; } ?> >0.0</option>
<option value="0.5" <?php if ($minimum == 0.5) { echo 'selected'; } ?> >0.5</option>
<option value="1.0" <?php if ($minimum == 1.0) { echo 'selected'; } ?> >1.0</option>
<option value="1.5" <?php if ($minimum == 1.5) { echo 'selected'; } ?> >1.5</option>
<option value="2.0" <?php if ($minimum == 2.0) { echo 'selected'; } ?> >2.0</option>
<option value="2.5" <?php if ($minimum == 2.5) { echo 'selected'; } ?> >2.5</option>
<option value="3.0" <?php if ($minimum == 3.0) { echo 'selected'; } ?> >3.0</option>
<option value="3.5" <?php if ($minimum == 3.5) { echo 'selected'; } ?> >3.5</option>
<option value="4.0" <?php if ($minimum == 4.0) { echo 'selected'; } ?> >4.0</option>
<option value="4.5" <?php if ($minimum == 4.5) { echo 'selected'; } ?> >4.5</option>
<option value="5.0" <?php if ($minimum == 5.0) { echo 'selected'; } ?> >5.0</option>
</select> 
 Minimum Delta Cutoff:
<select name="mindelta">
<option value="0.0" <?php if ($mindelta == 0.0) { echo 'selected'; } ?> >0.0</option>
<option value="0.1" <?php if ($mindelta == 0.1) { echo 'selected'; } ?> >0.1</option>
<option value="0.2" <?php if ($mindelta == 0.2) { echo 'selected'; } ?> >0.2</option>
<option value="0.3" <?php if ($mindelta == 0.3) { echo 'selected'; } ?> >0.3</option>
<option value="0.4" <?php if ($mindelta == 0.4) { echo 'selected'; } ?> >0.4</option>
<option value="0.5" <?php if ($mindelta == 0.5) { echo 'selected'; } ?> >0.5</option>
<option value="0.6" <?php if ($mindelta == 0.6) { echo 'selected'; } ?> >0.6</option>
<option value="0.7" <?php if ($mindelta == 0.7) { echo 'selected'; } ?> >0.7</option>
<option value="0.8" <?php if ($mindelta == 0.8) { echo 'selected'; } ?> >0.8</option>
<option value="0.9" <?php if ($mindelta == 0.9) { echo 'selected'; } ?> >0.9</option>
<option value="1.0" <?php if ($mindelta == 1.0) { echo 'selected'; } ?> >1.0</option>
</select> 
<select name="maptype">
<option value="magnitudes" <?php if ($maptype == 'magnitudes') { echo 'selected'; } ?> >magnitudes</option>
<option value="deltas" <?php if ($maptype == 'deltas') { echo 'selected'; } ?> >deltas</option>
<option value="intersections" <?php if ($maptype == 'intersections') { echo 'selected'; } ?> >intersections</option>
</select>
<input type="submit" value="Submit">
</form>
<?php
	if (($date = $_GET["date"])) {
		echo '<h1>' . $date . '</h1>';
		echo "<div id='map'></div><script>function initMap() { var map = new google.maps.Map(document.getElementById('map'), { zoom: 2, center: {lat: 0.0, lng: 0.0}, mapTypeId: 'terrain'});";
		$list_file = 'data/' . $model . '/list_' . $date . '.csv';
		if (($handle = fopen($list_file, 'r')) !== FALSE) {
			$handle = fopen($list_file, 'r');
			while (($data = fgetcsv($handle, 1000, ",")) !== FALSE) {
				$latitude = $data[0];
				$longitude = $data[1];
				if ($maptype == 'magnitudes') {
					$magnitude = number_format(round($data[2], 1), 1);
					if ($magnitude >= $minimum) {
						echo " var earthquake = new google.maps.Circle({";
						echo " strokeColor: 'yellow',";
						echo " fillColor: 'red',";
						echo " fillOpacity: " . $magnitude / 10.0 . ",";
						echo " map: map,";
						echo " center: {lat: " . $latitude . ", lng: " . $longitude . "},";
						echo " radius: 10000 * " . $magnitude . ",";
						echo " }); ";
						echo " var marker = new google.maps.Marker({position: {lat: " . $latitude . ", lng: " . $longitude . "}, label: '". $magnitude . "', title: '(" . $latitude . ", " . $longitude . ")', });";
						echo " marker.setMap(map);";
					}
				}
				if ($maptype == 'deltas') {
					$delta = number_format(round($data[3], 1), 1);
					if ($delta >= $mindelta) {
						echo " var earthquake = new google.maps.Circle({";
						echo " strokeColor: 'yellow',";
						echo " fillColor: 'red',";
						echo " fillOpacity: " . $delta . ",";
						echo " map: map,";
						echo " center: {lat: " . $latitude . ", lng: " . $longitude . "},";
						echo " radius: 10000 * " . $delta . ",";
						echo " }); ";
						echo " var marker = new google.maps.Marker({position: {lat: " . $latitude . ", lng: " . $longitude . "}, label: '+". $delta . "', title: '(" . $latitude . ", " . $longitude . ")', });";
						echo " marker.setMap(map);";
					}
				}
				if ($maptype == 'intersections') {
					$magnitude = number_format(round($data[2], 1), 1);
					$delta = number_format(round($data[3], 1), 1);
					if ($magnitude >= $minimum && $delta >= $mindelta) {
						echo " var earthquake = new google.maps.Circle({";
						echo " strokeColor: 'yellow',";
						echo " fillColor: 'red',";
						echo " fillOpacity: " . $magnitude / 10.0 . ",";
						echo " map: map,";
						echo " center: {lat: " . $latitude . ", lng: " . $longitude . "},";
						echo " radius: 10000 * " . $magnitude . ",";
						echo " }); ";
						echo " var marker = new google.maps.Marker({position: {lat: " . $latitude . ", lng: " . $longitude . "}, label: '". $magnitude . "(+" . $delta . ")" . "', title: '(" . $latitude . ", " . $longitude . ")', });";
						echo " marker.setMap(map);";
					}
				}
			}
		}
		fclose($handle);
		$key_file = fopen('key.txt', 'r');
		$key = fgets($key_file);
		fclose($key_file);
		echo '<script async defer src="https://maps.googleapis.com/maps/api/js?key=' . $key . '&callback=initMap" type="text/javascript"></script>';
		$map_file = 'images/' . $model . '/map_' . $date . '.png';
		if (file_exists($map_file)) {
			echo '<img src="' . $map_file . '">';
		} else {
			echo '<p>- NO MAP DATA -</p>';
		}
		$list_file = 'data/' . $model . '/list_' . $date . '.csv';
		if (($handle = fopen($list_file, 'r')) !== FALSE) {
			echo '<ul>';
			while (($data = fgetcsv($handle, 1000, ',')) !== FALSE) {
				echo "<li>Latitude: " . $data[0] . ", Longitude: " . $data[1] . ", Magnitude: " . number_format(round($data[2], 1), 1) . "</li>";
			}
			fclose($handle);
			echo '</ul>';
		} else {
			echo '<p>- NO LIST DATA -</p>';
		}
	} else {
		echo '<p>- PLEASE ENTER DATE -</p>';
	}
?>
<?php include("footer.php"); ?>
</body>
</html>
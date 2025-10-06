<?php
// predict.php

header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json");

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['error' => 'Método no permitido']);
    exit();
}

// Leer el cuerpo de la petición
$inputJSON = file_get_contents('php://input');
$input = json_decode($inputJSON, true);

// Validar campos mínimos
$requiredFields = [
    'fluorescence', 'chlorophylle', 'absorption', 'backscattering',
    'particulate_inorganic_carbon', 'diffuse_attenuation_coefficient',
    'remote_sensing_reflectance', 'temperature'
];

foreach ($requiredFields as $field) {
    if (!isset($input[$field])) {
        http_response_code(400);
        echo json_encode(['error' => "Falta el campo: $field"]);
        exit();
    }
}

// Mapear los nombres de PHP a los que espera Python
$pythonData = [[
    'nflh' => $input['fluorescence'],
    'chl' => $input['chlorophylle'],
    'a_412' => $input['absorption'],
    'bb_412' => $input['backscattering'],
    'pic' => $input['particulate_inorganic_carbon'],
    'kd_490' => $input['diffuse_attenuation_coefficient'],
    'rrs_412' => $input['remote_sensing_reflectance'],
    'sst' => $input['temperature']
]];

// Convertir a JSON para pasar a Python
$jsonArg = escapeshellarg(json_encode($pythonData));

// Ejecutar el script Python
$command = "python3 /var/www/sharktag.earth/predict_script.py --data $jsonArg 2>/dev/null";
exec($command, $output, $return_var);


// Unir salida y decodificar JSON
$outputStr = implode("", $output);
$result = json_decode($outputStr, true);

if ($return_var !== 0 || !isset($result['probabilities'])) {
    // Hubo un error ejecutando Python o no hay 'probabilities'
    http_response_code(500);
    echo json_encode(['error' => $result ?? $outputStr]);
    exit();
}

// Retornar solo el array de probabilities
echo json_encode([
    'probabilities' => $result['probabilities']
]);

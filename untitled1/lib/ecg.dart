import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:path/path.dart' as path;
import 'package:csv/csv.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:pdf/pdf.dart';
import 'package:path_provider/path_provider.dart';

class ECGPage extends StatefulWidget {
  @override
  _ECGPageState createState() => _ECGPageState();
}

class _ECGPageState extends State<ECGPage> {
  final _ageController = TextEditingController();
  final _tailleController = TextEditingController();
  final _poidsController = TextEditingController();
  String? _selectedSexe;
  File? _image;
  List<Map<String, dynamic>> _predictions = [];

  // Variable pour stocker le résultat de classification et le message à afficher
  Map<String, dynamic>? _classificationResult;
  String _classificationMessage = "";

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() => _image = File(picked.path));
    }
  }

  // Fonction pour générer le message de classification selon les critères fournis
  String _generateClassificationMessage(Map<String, dynamic> classification) {
    // On extrait la valeur de la classe "Normal (NORM)"
    bool isNormal = (classification["Normal (NORM)"] == 1);
    // On récupère les autres maladies dont la valeur vaut 1
    List<String> otherDiseases = [];
    classification.forEach((key, value) {
      if (key != "Normal (NORM)" && value == 1) {
        otherDiseases.add(key);
      }
    });

    // Cas où aucune maladie n'est détectée ou seule la classe "Normal (NORM)" est à 1
    if ((!isNormal && otherDiseases.isEmpty) ||
        (isNormal && otherDiseases.isEmpty)) {
      return "Cette ECG est normale, aucun risque détecté.";
    }
    // Cas où "Normal (NORM)" n'est pas présent et au moins une maladie détectée
    if (!isNormal && otherDiseases.isNotEmpty) {
      return "Le patient est risqué d'avoir un(e) ${otherDiseases.join(", ")}.";
    }
    // Cas où "Normal (NORM)" vaut 1 mais qu'il y a aussi d'autres maladies détectées
    if (isNormal && otherDiseases.isNotEmpty) {
      return "Cette ECG semble normale mais peut présenter un risque de ${otherDiseases.join(", ")}.";
    }
    // Par défaut, on retourne le message indiquant qu'aucun risque n'est détecté
    return "Cette ECG est normale, aucun risque détecté.";
  }

  // Fonction pour envoyer la requête à /predict et /classification
  Future<void> _sendPredictionRequest() async {
    if (_image == null ||
        _ageController.text.isEmpty ||
        _tailleController.text.isEmpty ||
        _poidsController.text.isEmpty ||
        _selectedSexe == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Remplis tous les champs et choisis une image")),
      );
      return;
    }

    // Préparation des données tabulaires
    final sexeValue = _selectedSexe == "Homme" ? 0 : 1;
    final tabularData =
        "${_ageController.text},$sexeValue,${_tailleController.text},${_poidsController.text}";

    // URL de base (à adapter à ton réseau)
    final baseUrl = "http://192.168.1.18:5000";

    // --- Appel à l'API de prédiction ---
    final predictUri = Uri.parse("$baseUrl/predict");
    var predictReq = http.MultipartRequest("POST", predictUri);
    predictReq.fields["tabular_data"] = tabularData;
    predictReq.files.add(
      await http.MultipartFile.fromPath(
        "image",
        _image!.path,
        contentType: MediaType(
          'image',
          path.extension(_image!.path).replaceAll('.', ''),
        ),
      ),
    );

    try {
      final predictStreamed = await predictReq.send();
      final predictBody = await predictStreamed.stream.bytesToString();
      if (predictStreamed.statusCode == 200) {
        // Parse JSON list des prédictions
        final List<dynamic> list = json.decode(predictBody);
        setState(() {
          _predictions =
              list
                  .map(
                    (e) => {
                      'Label': e['Label'],
                      'Valeur prédite':
                          e['Valeur pr\u00e9dite'] ?? e['Valeur prédite'],
                    },
                  )
                  .toList();
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Prédiction réussie")));
      } else {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Erreur API de prédiction")));
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Erreur de connexion")));
    }

    // --- Appel à l'API de classification ---
    final classifyUri = Uri.parse("$baseUrl/classification");
    var classifyReq = http.MultipartRequest("POST", classifyUri);
    classifyReq.fields["tabular_data"] = tabularData;
    classifyReq.files.add(
      await http.MultipartFile.fromPath(
        "image",
        _image!.path,
        contentType: MediaType(
          'image',
          path.extension(_image!.path).replaceAll('.', ''),
        ),
      ),
    );

    try {
      final classifyStreamed = await classifyReq.send();
      final classifyBody = await classifyStreamed.stream.bytesToString();
      if (classifyStreamed.statusCode == 200) {
        // On suppose que l'API retourne un map des résultats de classification
        final Map<String, dynamic> classification =
            json.decode(classifyBody) as Map<String, dynamic>;
        setState(() {
          _classificationResult = classification;
          _classificationMessage = _generateClassificationMessage(
            classification,
          );
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Classification réussie")));
      } else {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Erreur API de classification")));
      }
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Erreur de connexion")));
    }
  }

  Future<void> _exportCSV() async {
    final rows = [
      ['Label', 'Valeur prédite'],
      ..._predictions.map((p) => [p['Label'], p['Valeur prédite']]),
    ];
    final csv = const ListToCsvConverter().convert(rows);
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/predictions.csv');
    await file.writeAsString(csv);
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text("Exporté CSV")));
  }

  Future<void> _exportPDF() async {
    final pdf = pw.Document();
    final data =
        _predictions.map((p) => [p['Label'], p['Valeur prédite']]).toList();

    pdf.addPage(
      pw.Page(
        pageFormat: PdfPageFormat.a4,
        build:
            (_) => pw.Table.fromTextArray(
              headers: ['Label', 'Valeur prédite'],
              data: data,
              headerStyle: pw.TextStyle(
                color: PdfColors.white,
                fontWeight: pw.FontWeight.bold,
              ),
              headerDecoration: pw.BoxDecoration(color: PdfColors.blue),
            ),
      ),
    );

    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/predictions.pdf');
    await file.writeAsBytes(await pdf.save());
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text("Exporté PDF")));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("ECG"), backgroundColor: Color(0xff0095FF)),
      body: Padding(
        padding: EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            children: [
              // Inputs
              TextField(
                controller: _ageController,
                decoration: InputDecoration(labelText: "Âge"),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 8),
              DropdownButtonFormField<String>(
                value: _selectedSexe,
                decoration: InputDecoration(labelText: "Sexe"),
                items:
                    ['Homme', 'Femme']
                        .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                        .toList(),
                onChanged: (v) => setState(() => _selectedSexe = v),
              ),
              SizedBox(height: 8),
              TextField(
                controller: _tailleController,
                decoration: InputDecoration(labelText: "Taille (cm)"),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 8),
              TextField(
                controller: _poidsController,
                decoration: InputDecoration(labelText: "Poids (kg)"),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 16),
              // Image picker
              _image == null
                  ? ElevatedButton.icon(
                    onPressed: _pickImage,
                    icon: Icon(Icons.photo),
                    label: Text("Choisir une image"),
                  )
                  : Image.file(_image!, height: 200),
              SizedBox(height: 16),
              ElevatedButton(
                onPressed: _sendPredictionRequest,
                child: Text("Soumettre"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color(0xff0095FF),
                  padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                ),
              ),
              SizedBox(height: 24),
              // Affichage du message de classification
              if (_classificationResult != null) ...[
                Text(
                  "Résultat de classification",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                SizedBox(height: 8),
                Text(
                  _classificationMessage,
                  style: TextStyle(fontSize: 16, color: Colors.black87),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 16),
              ],
              // Affichage du tableau des prédictions
              if (_predictions.isNotEmpty) ...[
                Text(
                  "Résultats de la prédiction",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: DataTable(
                    headingRowColor: MaterialStateProperty.all(Colors.blue[50]),
                    columns: [
                      DataColumn(
                        label: Text(
                          'Label',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                      DataColumn(
                        label: Text(
                          'Valeur prédite',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                    rows:
                        _predictions
                            .map(
                              (p) => DataRow(
                                cells: [
                                  DataCell(Text(p['Label'].toString())),
                                  DataCell(
                                    Text(p['Valeur prédite'].toString()),
                                  ),
                                ],
                              ),
                            )
                            .toList(),
                  ),
                ),
                SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(onPressed: _exportCSV, child: Text("CSV")),
                    ElevatedButton(onPressed: _exportPDF, child: Text("PDF")),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}

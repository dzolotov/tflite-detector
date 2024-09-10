import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'TensorFlow Objects Detection'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

typedef ObjectInfo = (String, double);

class _MyHomePageState extends State<MyHomePage> {
  late Interpreter interpreter;

  List<ObjectInfo> objects = [];

  // Uint8List? image;
  ui.Image? rawImage;

  @override
  void didChangeDependencies() async {
    super.didChangeDependencies();
    interpreter =
        await Interpreter.fromAsset('assets/mobilenet_v1_1.0_224.tflite');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // if (image != null) Image.memory(image!),
            if (rawImage != null)
              RawImage(
                image: rawImage!,
              ),
            ElevatedButton(
              onPressed: () async {
                //choose image from gallery
                final picker = ImagePicker();
                final camera =
                    await picker.pickImage(source: ImageSource.gallery);
                Uint8List? image = await camera?.readAsBytes();
                //resize to 224x224 (for mobilenet 1.0_224) and reshape vector
                final decoded = img.decodeImage(image!.toList());
                final input = img.copyResize(decoded!, width: 224, height: 224);
                final vector = input.getBytes(format: img.Format.rgb);
                final labels =
                    (await rootBundle.loadString('assets/labels.txt'))
                        .split('\n');
                //input vector is 1 (batch) x 224 (width) x 224 (height) x 3 (channels)
                final data = vector
                    .map((e) => e / 255)
                    .toList()
                    .reshape([1, 224, 224, 3]);
                //prepare output vector (1001 labels)
                final output = List.filled(1001, 0).reshape([1, 1001]);

                interpreter.run(data, output);
                //search for maximum values
                List<double> probs = output.first;
                final indexed = probs.indexed.toList();
                //sort by value, but save the keys (indexed in labels.txt)
                indexed.sort((a, b) => b.$2.compareTo(a.$2));
                final top5 = indexed.sublist(0, 5);
                objects.clear();

                //programmatically create picture
                final recorder = ui.PictureRecorder();
                final canvas = Canvas(
                  recorder,
                  Offset.zero &
                      Size(
                        decoded.width.toDouble(),
                        decoded.height.toDouble(),
                      ),
                );
                //convert image to ui.Image
                ui.decodeImageFromList(image, (image) async {
                  //draw as background
                  canvas.drawImage(image, ui.Offset.zero, ui.Paint());
                  //draw probabilities over picture
                  for (final (index, entry) in top5.indexed) {
                    final paragraph = TextPainter(
                      textScaler: const TextScaler.linear(3.0),
                      textDirection: ui.TextDirection.ltr,
                      text: TextSpan(
                        text:
                            '${labels[entry.$1]}: ${entry.$2.toStringAsFixed(2)}',
                        style: const TextStyle(color: Colors.red),
                      ),
                    );
                    paragraph.layout();
                    paragraph.paint(canvas, ui.Offset(0, index * 48));
                    objects.add((labels[entry.$1], entry.$2));
                  }
                  //extract raw image (for with RawImage)
                  final picture = recorder.endRecording();
                  rawImage =
                      await picture.toImage(decoded.width, decoded.height);
                  setState(() {});
                });
              },
              child: const Text('Choose image'),
            ),
          ],
        ),
      ),
    );
  }
}

import 'dart:io';
import 'dart:typed_data';

import 'package:facial_recognition/pick_image.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as img;
import 'package:tflite/tflite.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  XFile? file;
  List<dynamic> list = [];
  final imagePicker = PickImage();

  Future<void> runModel() async {
    final res = await Tflite.loadModel(
  model: 'assets/new_model.tflite',
  // labels: "assets/labels.txt",
  // numThreads: 1, // defaults to 1
  // isAsset: true, // defaults to true, set to false to load resources outside assets
  // useGpuDelegate: false // defaults to false, set to true to use GPU delegate
);


    final imageBytes = await file!.readAsBytes();
    final image = img.decodeImage(imageBytes);
    final resized = img.copyResize(image!, width: 256, height: 256);
    final resizedImg = Uint8List.fromList(img.encodePng(resized));
// resize the image to 256 x 256
// For ex: if input tensor shape [1,5] and type is float32
    var input = resizedImg.map((e) => e / 255).toList();

var recognitions = await Tflite.runModelOnImage(
  path: file!.path,
  // imageMean: 127.5,   // defaults to 127.5
  // imageStd: 127.5,    // defaults to 127.5
  // rotation: 90,       // defaults to 90, Android only
  numResults: 11,      // defaults to 5
  // threshold: 0.1,     // defaults to 0.1
  // asynch: true        // defaults to true
);

    // final interpreter =
    //     await tfl.Interpreter.fromAsset('assets/new_model.tflite');

//     final imageBytes = await file!.readAsBytes();
//     final image = img.decodeImage(imageBytes);
//     final resized = img.copyResize(image!, width: 256, height: 256);
//     final resizedImg = Uint8List.fromList(img.encodePng(resized));
// // resize the image to 256 x 256
// // For ex: if input tensor shape [1,5] and type is float32
//     var input = resizedImg.map((e) => e / 255).toList();

// if output tensor shape [1,2] and type is float32
    // var output = [];

// inference
    // interpreter.run(input, output);

    print(recognitions);
    // list = output;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Face Detection'),
        actions: [
          ElevatedButton.icon(
            onPressed: () {
              runModel();
            },
            icon: Icon(Icons.code),
            label: Text('Run'),
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            if (file != null)
              Image.file(
                File(file!.path),
                fit: BoxFit.cover,
              ),
            ...list.map((e) => Text(e.toString())),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          showModalBottomSheet(
              isDismissible: true,
              context: context,
              builder: (context) {
                return Container(
                  height: 100,
                  width: double.infinity,
                  decoration: const BoxDecoration(
                      borderRadius: BorderRadius.only(
                          topLeft: Radius.circular(15),
                          topRight: Radius.circular(15))),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      Column(
                        children: [
                          IconButton(
                              onPressed: () async {
                                Navigator.pop(context);
                                file = await imagePicker.pickImageFromCamera();
                                setState(() {});
                              },
                              icon: const Icon(
                                Icons.camera,
                                // color: AppColors.brown,
                                size: 30,
                              )),
                          Text(
                            "Camera",
                            // style: AppStyles.text,
                          )
                        ],
                      ),
                      Column(
                        children: [
                          IconButton(
                              onPressed: () async {
                                Navigator.pop(context);
                                file = await imagePicker.pickImageFromGallery();
                                setState(() {});
                              },
                              icon: const Icon(
                                Icons.photo,
                                // color: AppColors.brown,
                                size: 30,
                              )),
                          Text(
                            "Gallery",
                            // style: AppStyles.text,
                          )
                        ],
                      )
                    ],
                  ),
                );
              });
        },
        child: Icon(Icons.upload_rounded),
      ),
    );
  }
}

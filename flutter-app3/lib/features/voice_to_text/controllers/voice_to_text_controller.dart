import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:logger/logger.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'package:ai_voice_to_hand_signs_project/data/models/session.dart';
import 'package:ai_voice_to_hand_signs_project/data/models/transcript_entry.dart';
import 'package:ai_voice_to_hand_signs_project/data/services/cloud_functions_service.dart';
import 'package:ai_voice_to_hand_signs_project/data/services/database_service.dart';

class VoiceToTextController extends GetxController {
  final Logger _logger = Logger();
  final DatabaseService _dbService = Get.find<DatabaseService>();
  final CloudFunctionsService _cfService = Get.find<CloudFunctionsService>();

  // Audio recording & playback
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();
  
  // Observable state
  final RxBool isRecording = false.obs;
  final RxBool isProcessing = false.obs;
  final RxString transcribedText = ''.obs;
  final RxString statusMessage = 'Ready'.obs;
  final RxList<TranscriptEntry> transcriptHistory = <TranscriptEntry>[].obs;

  String? _currentSessionId;
  String? _tempAudioPath;

  @override
  void onInit() {
    super.onInit();
    _logger.i('VoiceToTextController initialized');
  }

  @override
  void onClose() {
    _recorder.dispose();
    _audioPlayer.dispose();
    super.onClose();
  }

  /// Start voice recording
  Future<void> startRecording() async {
    try {
      if (await _recorder.hasPermission()) {
        final tempDir = await getTemporaryDirectory();
        _tempAudioPath = p.join(tempDir.path, 'voice_record.m4a');

        // Start session if not already active
        if (_currentSessionId == null) {
          final user = FirebaseAuth.instance.currentUser;
          if (user != null) {
            _currentSessionId = await _dbService.createSession(
              user.uid,
              SessionType.voiceToSign,
            );
          }
        }

        await _recorder.start(const RecordConfig(), path: _tempAudioPath!);
        isRecording.value = true;
        statusMessage.value = 'Recording...';
        _logger.i('Recording started at $_tempAudioPath');
      } else {
        statusMessage.value = 'Microphone permission denied';
      }
    } catch (e) {
      _logger.e('Error starting recording: $e');
      statusMessage.value = 'Error starting recording';
    }
  }

  /// Stop recording and process STT
  Future<void> stopRecording() async {
    try {
      final path = await _recorder.stop();
      isRecording.value = false;
      
      if (path != null) {
        statusMessage.value = 'Processing speech...';
        await _processSTT(path);
      }
    } catch (e) {
      _logger.e('Error stopping recording: $e');
      statusMessage.value = 'Error stopping recording';
    }
  }

  /// Process Speech-to-Text via Cloud Functions
  Future<void> _processSTT(String path) async {
    isProcessing.value = true;
    try {
      final file = File(path);
      final bytes = await file.readAsBytes();
      final base64Audio = base64Encode(bytes);

      // Call STT Cloud Function
      // Note: We're using m4a/aac, so encoding might need adjustment in backend
      // or here if LINEAR16 is strictly required. 
      // Defaulting to what's in service for now.
      final text = await _cfService.speechToText(
        base64Audio,
        lang: 'en-US',
        encoding: 'ENCODING_UNSPECIFIED', // Let Google Cloud detect
      );

      if (text.isNotEmpty) {
        transcribedText.value = text;
        statusMessage.value = 'Transcribed';
        
        // Save to RTDB
        if (_currentSessionId != null) {
          final entry = TranscriptEntry(
            type: TranscriptType.sttText, // Corrected from stt_text
            content: text,
            timestamp: DateTime.now().millisecondsSinceEpoch,
            speakerRole: SpeakerRole.lecturer,
          );
          await _dbService.addTranscriptEntry(_currentSessionId!, entry);
          transcriptHistory.insert(0, entry);
        }
      } else {
        statusMessage.value = 'No speech detected';
      }
    } catch (e) {
      _logger.e('STT processing error: $e');
      statusMessage.value = 'STT Error: ${e.toString()}';
    } finally {
      isProcessing.value = false;
    }
  }

  /// Text-to-Speech: Speak the current transcribed text (or manual input)
  Future<void> speakText(String text) async {
    if (text.isEmpty) return;
    
    isProcessing.value = true;
    statusMessage.value = 'Generating speech...';
    
    try {
      final base64Audio = await _cfService.textToSpeech(text, lang: 'en');
      
      if (base64Audio.isNotEmpty) {
        final bytes = base64Decode(base64Audio);
        await _audioPlayer.play(BytesSource(Uint8List.fromList(bytes)));
        statusMessage.value = 'Playing...';
      }
    } catch (e) {
      _logger.e('TTS error: $e');
      statusMessage.value = 'TTS Error';
    } finally {
      isProcessing.value = false;
    }
  }

  /// Manually update transcribed text (e.g. user correction)
  void updateText(String text) {
    transcribedText.value = text;
  }

  /// Clear session and history
  void clearAll() {
    transcribedText.value = '';
    transcriptHistory.clear();
    statusMessage.value = 'Ready';
  }
}

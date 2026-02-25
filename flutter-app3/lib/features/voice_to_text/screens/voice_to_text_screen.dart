import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:iconsax_flutter/iconsax_flutter.dart';
import 'package:ai_voice_to_hand_signs_project/features/voice_to_text/controllers/voice_to_text_controller.dart';
import 'package:ai_voice_to_hand_signs_project/util/constants/colors.dart';

class VoiceToTextScreen extends StatelessWidget {
  const VoiceToTextScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.put(VoiceToTextController());
    final TextEditingController textController = TextEditingController();

    // Sync TextEditingController with RxString
    ever(controller.transcribedText, (String value) {
      if (textController.text != value) {
        textController.text = value;
      }
    });

    return Scaffold(
      backgroundColor: TColors.darkBackground,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          "Voice to Text",
          style: TextStyle(color: TColors.textPrimary),
        ),
        leading: IconButton(
          icon: const Icon(Iconsax.arrow_left, color: TColors.textPrimary),
          onPressed: () => Get.back(),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Status Indicator
            Obx(() => Text(
                  controller.statusMessage.value,
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    color: controller.isRecording.value ? TColors.error : TColors.textSecondary,
                    fontWeight: FontWeight.w500,
                  ),
                )),
            const SizedBox(height: 20),

            // Transcript Card
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: TColors.darkContainer,
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: TColors.grey.withAlpha(50)),
                ),
                child: Column(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: textController,
                        maxLines: null,
                        style: const TextStyle(
                          color: TColors.textPrimary,
                          fontSize: 18,
                          height: 1.5,
                        ),
                        decoration: const InputDecoration(
                          hintText: "Transcription will appear here...",
                          hintStyle: TextStyle(color: TColors.textSecondary),
                          border: InputBorder.none,
                        ),
                        onChanged: (value) => controller.updateText(value),
                      ),
                    ),
                    if (controller.isProcessing.value)
                      const LinearProgressIndicator(
                        backgroundColor: Colors.transparent,
                        color: TColors.primary,
                      ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 30),

            // Control Buttons
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                // Speak Button (TTS)
                _buildActionButton(
                  icon: Iconsax.volume_high,
                  label: "Speak",
                  onTap: () => controller.speakText(textController.text),
                  color: TColors.primary,
                ),

                // Record Button (STT)
                Obx(() => _buildRecordButton(
                      isRecording: controller.isRecording.value,
                      onTap: () {
                        if (controller.isRecording.value) {
                          controller.stopRecording();
                        } else {
                          controller.startRecording();
                        }
                      },
                    )),

                // Clear Button
                _buildActionButton(
                  icon: Iconsax.trash,
                  label: "Clear",
                  onTap: () {
                    controller.clearAll();
                    textController.clear();
                  },
                  color: TColors.error,
                ),
              ],
            ),
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    required Color color,
  }) {
    return Column(
      children: [
        GestureDetector(
          onTap: onTap,
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: color.withAlpha(30),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: color.withAlpha(50)),
            ),
            child: Icon(icon, color: color, size: 28),
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(color: TColors.textSecondary, fontSize: 12),
        ),
      ],
    );
  }

  Widget _buildRecordButton({
    required bool isRecording,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          color: isRecording ? TColors.error : TColors.primary,
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: (isRecording ? TColors.error : TColors.primary).withAlpha(100),
              blurRadius: 15,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Icon(
          isRecording ? Iconsax.stop : Iconsax.microphone,
          color: Colors.white,
          size: 40,
        ),
      ),
    );
  }
}

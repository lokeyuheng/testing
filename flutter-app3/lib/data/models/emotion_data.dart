enum EmotionType {
  happy,
  angry,
  down,
  confused,
  questioning,
  neutral;

  String get label {
    switch (this) {
      case EmotionType.happy:
        return 'Happy';
      case EmotionType.angry:
        return 'Angry';
      case EmotionType.down:
        return 'Sad/Down';
      case EmotionType.confused:
        return 'Confused';
      case EmotionType.questioning:
        return 'Questioning';
      case EmotionType.neutral:
        return 'Neutral';
    }
  }

  String get emoji {
    switch (this) {
      case EmotionType.happy:
        return '😊';
      case EmotionType.angry:
        return '😠';
      case EmotionType.down:
        return '😔';
      case EmotionType.confused:
        return '😕';
      case EmotionType.questioning:
        return '🤔';
      case EmotionType.neutral:
        return '😐';
    }
  }
}

class EmotionData {
  final EmotionType emotion;
  final double confidence;
  final int timestamp;

  EmotionData({
    required this.emotion,
    required this.confidence,
    required this.timestamp,
  });

  String get label => emotion.label;
  String get emoji => emotion.emoji;
}

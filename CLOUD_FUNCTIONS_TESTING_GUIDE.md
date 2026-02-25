# Cloud Storage & Cloud Functions — Manual Testing Guide

## Prerequisites

- ✅ Firebase CLI installed (`firebase --version`)
- ✅ Logged in to Firebase (`firebase login`)
- ✅ Blaze plan enabled
- ✅ GCP APIs enabled (Cloud Functions, Vertex AI, STT, TTS, Translation)
- ✅ TypeScript compiles (`cd functions && npm run build`)

---

## Step 1: Deploy Cloud Functions

```bash
cd flutter-app
firebase deploy --only functions
```

This will deploy all 7 functions to `asia-southeast1`:
- `onUserCreated` (Auth trigger)
- `glossToSentence` (HTTPS Callable)
- `sentenceToGloss` (HTTPS Callable)
- `lookupSignVideos` (HTTPS Callable)
- `speechToText` (HTTPS Callable)
- `textToSpeech` (HTTPS Callable)
- `translateText` (HTTPS Callable)

### Deploy Storage & Database Rules (optional — if not done yet)
```bash
firebase deploy --only storage
firebase deploy --only database
```

---

## Step 2: Test `onUserCreated` (Auth Trigger)

This triggers automatically when a new user signs up.

### How to test:
1. Open your Flutter app
2. Create a **new account** (email/password or Google Sign-In)
3. Open [Firebase Console → Realtime Database](https://console.firebase.google.com/project/ai-real-time-voice-to-sign/database)
4. Check that `/users/<your-uid>` was created with:
   ```json
   {
     "displayName": "Your Name",
     "email": "your@email.com",
     "photoUrl": "...",
     "preferredLanguage": "en",
     "createdAt": 1707600000000,
     "lastActiveAt": 1707600000000
   }
   ```

### Expected result:
✅ User profile node auto-created in RTDB

---

## Step 3: Test `glossToSentence` (Gemini)

Converts sign language gloss words into a natural sentence.

### Option A: Firebase Shell
```bash
cd flutter-app
firebase functions:shell
```
Then in the shell:
```js
glossToSentence({data: {glossWords: ["HELLO", "HOW", "YOU"], lang: "en"}, auth: {uid: "test-user"}})
```

### Option B: curl (after deployment)
```bash
# Get your ID token first from Firebase Auth
curl -X POST \
  https://asia-southeast1-ai-real-time-voice-to-sign.cloudfunctions.net/glossToSentence \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ID_TOKEN" \
  -d '{"data": {"glossWords": ["HELLO", "HOW", "YOU"], "lang": "en"}}'
```

### Option C: From Flutter app
```dart
final cfService = Get.find<CloudFunctionsService>();
final sentence = await cfService.glossToSentence(["HELLO", "HOW", "YOU"]);
print(sentence); // e.g., "Hello, how are you?"
```

### Expected result:
✅ `{"result": {"sentence": "Hello, how are you?"}}` (or similar natural sentence)

---

## Step 4: Test `sentenceToGloss` (Gemini)

Converts a natural sentence into sign language gloss words.

### Firebase Shell:
```js
sentenceToGloss({data: {sentence: "How are you doing today?", lang: "en"}, auth: {uid: "test-user"}})
```

### From Flutter:
```dart
final glossWords = await cfService.sentenceToGloss("How are you doing today?");
print(glossWords); // e.g., ["HOW", "YOU", "TODAY"]
```

### Expected result:
✅ `{"result": {"glossWords": ["HOW", "YOU", "TODAY"]}}` (or similar)

---

## Step 5: Test `translateText` (Google Translate)

### Firebase Shell:
```js
translateText({data: {text: "Hello, how are you?", targetLang: "ms"}, auth: {uid: "test-user"}})
```

### From Flutter:
```dart
final translated = await cfService.translateText(
  "Hello, how are you?",
  targetLang: "ms",
);
print(translated); // "Helo, apa khabar?"
```

### Test with different languages:
```js
// Japanese
translateText({data: {text: "Hello", targetLang: "ja"}, auth: {uid: "test-user"}})
// Chinese
translateText({data: {text: "Thank you", targetLang: "zh"}, auth: {uid: "test-user"}})
// Korean
translateText({data: {text: "Good morning", targetLang: "ko"}, auth: {uid: "test-user"}})
```

### Expected result:
✅ Translated text in the target language

---

## Step 6: Test `textToSpeech` (Google Cloud TTS)

### Firebase Shell:
```js
textToSpeech({data: {text: "Hello, how are you?", lang: "en"}, auth: {uid: "test-user"}})
```

### From Flutter:
```dart
final audioBase64 = await cfService.textToSpeech("Hello, how are you?", lang: "en");
// Decode base64 and play with audioplayers package
```

### Expected result:
✅ `{"result": {"audioBase64": "...(long base64 string)...", "contentType": "audio/mp3"}}`

---

## Step 7: Test `speechToText` (Google Cloud STT)

This requires base64-encoded audio. For testing:

### From Flutter (record audio first):
```dart
// After recording audio to a file:
final bytes = await File('recording.wav').readAsBytes();
final audioBase64 = base64Encode(bytes);
final text = await cfService.speechToText(audioBase64, lang: "en-US");
print(text); // Transcribed text
```

### Expected result:
✅ Transcribed text from speech audio

---

## Step 8: Test `lookupSignVideos` (RTDB + Storage)

This requires sign videos to be uploaded first.

### Step 8a: Upload test videos to Cloud Storage

Using the Firebase Console or gsutil:
```bash
# Upload a test sign video
gsutil cp hello.mp4 gs://ai-real-time-voice-to-sign.firebasestorage.app/signs/hello.mp4

# Upload fingerspelling videos
gsutil cp a.mp4 gs://ai-real-time-voice-to-sign.firebasestorage.app/fingerspelling/a.mp4
gsutil cp b.mp4 gs://ai-real-time-voice-to-sign.firebasestorage.app/fingerspelling/b.mp4
```

Or upload via [Firebase Console → Storage](https://console.firebase.google.com/project/ai-real-time-voice-to-sign/storage).

### Step 8b: Seed the RTDB catalog

Manually add entries in Firebase Console → Realtime Database:

```json
{
  "signVideoCatalog": {
    "hello": {
      "videoUrl": "gs://ai-real-time-voice-to-sign.firebasestorage.app/signs/hello.mp4",
      "thumbnailUrl": "gs://ai-real-time-voice-to-sign.firebasestorage.app/thumbnails/hello.jpg",
      "duration": 2.5,
      "addedAt": 1707600000000
    }
  },
  "fingerspelling": {
    "a": {
      "videoUrl": "gs://ai-real-time-voice-to-sign.firebasestorage.app/fingerspelling/a.mp4",
      "duration": 1.0
    }
  }
}
```

### Step 8c: Test the function

Firebase Shell:
```js
lookupSignVideos({data: {glossWords: ["hello", "xyz"]}, auth: {uid: "test-user"}})
```

### Expected result:
✅ `hello` → returns sign video URL
✅ `xyz` → falls back to fingerspelling (`x`, `y`, `z` individual letter videos)

---

## Step 9: Check Cloud Function Logs

After testing, verify all functions executed correctly:

```bash
firebase functions:log --only glossToSentence
firebase functions:log --only sentenceToGloss
firebase functions:log --only translateText
firebase functions:log --only onUserCreated
```

Or view in [Google Cloud Console → Cloud Functions → Logs](https://console.cloud.google.com/functions/list?project=ai-real-time-voice-to-sign)

---

## Step 10: End-to-End Flutter Test

1. Run the Flutter app
2. Login (triggers `onUserCreated` → check RTDB)
3. Navigate to Sign-to-Voice screen
4. Start recording → detected words are saved to RTDB
5. In code or via debug button, call:
   ```dart
   final cfService = Get.find<CloudFunctionsService>();
   
   // Test gloss → sentence
   final sentence = await cfService.glossToSentence(["HELLO", "HOW", "YOU"]);
   
   // Test sentence → gloss
   final gloss = await cfService.sentenceToGloss("How are you?");
   
   // Test translation
   final translated = await cfService.translateText("Hello", targetLang: "ms");
   
   // Test TTS
   final audio = await cfService.textToSpeech("Hello", lang: "en");
   
   // Test video lookup
   final videos = await cfService.lookupSignVideos(["HELLO"]);
   ```

---

## Troubleshooting

### "Permission denied" errors
- Ensure your Firebase project is on the Blaze plan
- Enable the required GCP APIs
- Check that the Cloud Functions service account has the necessary roles

### "Function not found" errors
- Verify deployment: `firebase functions:list`
- Ensure region matches (`asia-southeast1`) in both Cloud Functions and Flutter client

### "Unauthenticated" errors
- Functions require an authenticated user
- In Firebase Shell, always pass `auth: {uid: "test-user"}`
- In Flutter, ensure user is logged in before calling

### Slow cold starts
- First invocation may take 5-10 seconds (cold start)
- Subsequent calls within ~15 minutes will be much faster
- Consider setting `minInstances: 1` for critical functions (costs more)

---

## Files Created

| File | Purpose |
|------|---------|
| `storage.rules` | Cloud Storage security rules |
| `firebase.json` | Updated with functions, storage, database config |
| `functions/package.json` | NPM dependencies |
| `functions/tsconfig.json` | TypeScript config |
| `functions/.eslintrc.js` | Linting config |
| `functions/.gitignore` | Git ignore for node_modules, lib |
| `functions/src/index.ts` | Entry point exporting all functions |
| `functions/src/auth/onUserCreated.ts` | Auth trigger → RTDB profile |
| `functions/src/ai/glossToSentence.ts` | Gemini: gloss → sentence |
| `functions/src/ai/sentenceToGloss.ts` | Gemini: sentence → gloss |
| `functions/src/ai/speechToText.ts` | Cloud STT |
| `functions/src/ai/textToSpeech.ts` | Cloud TTS |
| `functions/src/translation/translateText.ts` | Cloud Translation |
| `functions/src/signs/lookupSignVideos.ts` | RTDB video lookup + fallback |
| `functions/src/seed/seedVideoCatalog.ts` | Admin: seed RTDB from Storage |
| `lib/data/services/cloud_functions_service.dart` | Flutter client for callables |

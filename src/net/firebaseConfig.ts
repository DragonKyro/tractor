// The Firebase project online rooms live in. Public by design: a web app's
// Firebase config only identifies the project; access is decided by the
// security rules in database.rules.json.
//
// The "Tractor" web app registered in github-games-e7c7d, a project shared
// with the other game apps (only the Realtime Database and anonymous Auth
// are used; this app's data lives under /tractor). Blank the projectId to
// switch online play off (isOnlineConfigured() is then false and the UI
// hides it).

export const FIREBASE_CONFIG = {
  apiKey: 'AIzaSyAmAirzwcz926PyKh-bfUUIn7OSkcve6tI',
  authDomain: 'github-games-e7c7d.firebaseapp.com',
  databaseURL: 'https://github-games-e7c7d-default-rtdb.firebaseio.com',
  projectId: 'github-games-e7c7d',
  appId: '1:879254296864:web:6cc3fb8b4fa2acf18f1f45',
};

export function isOnlineConfigured(): boolean {
  return FIREBASE_CONFIG.projectId !== '' && FIREBASE_CONFIG.databaseURL !== '' && FIREBASE_CONFIG.apiKey !== '';
}

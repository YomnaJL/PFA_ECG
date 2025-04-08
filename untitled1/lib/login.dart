import 'package:CardioAI/Modele.dart';
import 'package:flutter/material.dart';
import 'package:CardioAI/signup.dart'; // Assurez-vous d'importer la page de création de compte
import 'package:fluttertoast/fluttertoast.dart';
import 'package:CardioAI/ecg.dart'; // Assurez-vous d'importer la page ECG
import 'package:shared_preferences/shared_preferences.dart';

class LoginPage extends StatefulWidget {
  @override
  _LoginPageState createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();

  // Méthode de connexion
  void _login() async {
    final email = _emailController.text;
    final password = _passwordController.text;

    if (email.isEmpty || password.isEmpty) {
      Fluttertoast.showToast(
        msg: "Veuillez remplir tous les champs",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.BOTTOM,
        timeInSecForIosWeb: 1,
        backgroundColor: Colors.red,
        textColor: Colors.white,
        fontSize: 16.0,
      );
      return;
    }

    // Récupérer les informations stockées dans SharedPreferences
    SharedPreferences prefs = await SharedPreferences.getInstance();
    String? storedEmail = prefs.getString('email');
    String? storedPassword = prefs.getString('password');

    if (storedEmail == email && storedPassword == password) {
      // Si le compte existe, rediriger vers la page ECG avec les deux modèles
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => WelcomeScreen()),
      );
    } else {
      Fluttertoast.showToast(
        msg: "Identifiants incorrects",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.BOTTOM,
        timeInSecForIosWeb: 1,
        backgroundColor: Colors.red,
        textColor: Colors.white,
        fontSize: 16.0,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.white,
      body: Stack(
        children: [
          // ✅ Fond dégradé
          Container(
            width: double.infinity,
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                colors: [Color(0xff0095FF), Color(0xff00D4FF)],
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
              ),
            ),
          ),

          // ✅ Contenu principal
          Column(
            children: [
              const SizedBox(height: 50),

              // 🔙 Bouton retour
              Align(
                alignment: Alignment.centerLeft,
                child: IconButton(
                  icon: const Icon(
                    Icons.arrow_back,
                    color: Colors.white,
                    size: 30,
                  ),
                  onPressed: () {
                    Navigator.pop(context);
                  },
                ),
              ),

              const SizedBox(height: 10),
              const Text(
                "Bienvenue!",
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              const Text(
                "Veuillez connecter pour continuer !",
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
              const SizedBox(height: 40),

              // ✅ Formulaire de connexion
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 30),
                  decoration: const BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.vertical(
                      top: Radius.circular(30),
                    ),
                  ),
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      inputFile(
                        label: "Email",
                        icon: Icons.email,
                        controller: _emailController,
                      ),
                      inputFile(
                        label: "mot de passe",
                        icon: Icons.lock,
                        obscureText: true,
                        controller: _passwordController,
                      ),
                      const SizedBox(height: 10),

                      // 🔵 Bouton Login
                      MaterialButton(
                        minWidth: double.infinity,
                        height: 55,
                        onPressed: _login, // Appel de la méthode de connexion
                        color: const Color(0xff0095FF),
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(50),
                        ),
                        child: const Text(
                          "Login",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(height: 15),

                      // 🔵 Lien vers Sign Up
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          const Text("avez vous un compte ? "),
                          GestureDetector(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => SignUp(),
                                ),
                              );
                            },
                            child: const Text(
                              "inscrisez-vous",
                              style: TextStyle(
                                fontWeight: FontWeight.w600,
                                fontSize: 18,
                                color: Color(0xff0095FF),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),

                      // 🔵 Image (en bas)
                      Container(
                        height: 180,
                        decoration: const BoxDecoration(
                          image: DecorationImage(
                            image: AssetImage("assets/Login.png"),
                            fit: BoxFit.contain,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ✅ Widget personnalisé pour les champs de saisie
Widget inputFile({
  required String label,
  required IconData icon,
  bool obscureText = false,
  required TextEditingController controller, // Ajout du contrôleur de texte
}) {
  return Padding(
    padding: const EdgeInsets.only(bottom: 20),
    child: TextField(
      controller: controller, // Lier le contrôleur à ce champ
      obscureText: obscureText,
      decoration: InputDecoration(
        labelText: label,
        prefixIcon: Icon(icon, color: Colors.blue),
        filled: true,
        fillColor: Colors.grey[200],
        contentPadding: const EdgeInsets.symmetric(
          vertical: 18,
          horizontal: 20,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(30),
          borderSide: BorderSide.none,
        ),
      ),
    ),
  );
}

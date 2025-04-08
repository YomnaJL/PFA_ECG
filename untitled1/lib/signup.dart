import 'package:flutter/material.dart';
import 'package:CardioAI/login.dart'; // Assurez-vous d'importer Login
import 'package:fluttertoast/fluttertoast.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SignUp extends StatefulWidget {
  @override
  _SignUpState createState() => _SignUpState();
}

class _SignUpState extends State<SignUp> {
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();

  // Validation de l'email
  bool _isValidEmail(String email) {
    RegExp emailRegex = RegExp(r'^[^@]+@[^@]+\.[^@]+');
    return emailRegex.hasMatch(email);
  }

  // Validation des champs
  bool _validateFields() {
    if (_emailController.text.isEmpty ||
        _passwordController.text.isEmpty ||
        _confirmPasswordController.text.isEmpty) {
      Fluttertoast.showToast(
        msg: "Tous les champs doivent être remplis",
        toastLength: Toast.LENGTH_LONG,
        gravity: ToastGravity.BOTTOM,
        timeInSecForIosWeb: 7,
        backgroundColor: Colors.blue,
        textColor: Colors.white,
        fontSize: 19.0,
      );
      return false;
    }

    if (!_isValidEmail(_emailController.text)) {
      Fluttertoast.showToast(
        msg: "Veuillez entrer un email valide",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.BOTTOM,
        timeInSecForIosWeb: 1,
        backgroundColor: Colors.blue,
        textColor: Colors.white,
        fontSize: 16.0,
      );
      return false;
    }

    if (_passwordController.text != _confirmPasswordController.text) {
      Fluttertoast.showToast(
        msg: "Les mots de passe ne correspondent pas",
        toastLength: Toast.LENGTH_SHORT,
        gravity: ToastGravity.BOTTOM,
        timeInSecForIosWeb: 1,
        backgroundColor: Colors.blue,
        textColor: Colors.white,
        fontSize: 16.0,
      );
      return false;
    }

    return true;
  }

  // Méthode pour enregistrer le compte et rediriger vers la page Login
  void _saveAccount() async {
    if (_validateFields()) {
      final email = _emailController.text;
      final password = _passwordController.text;

      try {
        // Sauvegarder les informations dans SharedPreferences
        SharedPreferences prefs = await SharedPreferences.getInstance();
        await prefs.setString('email', email);
        await prefs.setString('password', password);

        // Affichage du Toast de succès
        Fluttertoast.showToast(
          msg: "Compte créé avec succès !",
          toastLength: Toast.LENGTH_LONG,
          gravity: ToastGravity.CENTER,
          timeInSecForIosWeb: 1,
          backgroundColor: Colors.green,
          textColor: Colors.white,
          fontSize: 20.0,
        );

        // Attendre quelques secondes avant de rediriger
        await Future.delayed(Duration(seconds: 2));

        // Rediriger vers la page Login après l'enregistrement
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => LoginPage()),
        );
      } catch (e, stacktrace) {
        // Affichage du Toast en cas d'erreur
        Fluttertoast.showToast(
          msg: "Erreur lors de la sauvegarde du compte: ${e.toString()}",
          toastLength: Toast.LENGTH_LONG,
          gravity: ToastGravity.CENTER,
          timeInSecForIosWeb: 1,
          backgroundColor: Colors.blue,
          textColor: Colors.white,
          fontSize: 20.0,
        );
        print("Erreur lors de la sauvegarde du compte: $e");
        print("Stacktrace: $stacktrace");
      }
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
                "créer un compte",
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              const Text(
                " Bienvenue dans CardioAI !",
                style: TextStyle(fontSize: 18, color: Colors.white70),
              ),
              const SizedBox(height: 40),

              // ✅ Formulaire de création de compte
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
                        label: "Password",
                        icon: Icons.lock,
                        obscureText: true,
                        controller: _passwordController,
                      ),
                      inputFile(
                        label: "Confirm Password",
                        icon: Icons.lock,
                        obscureText: true,
                        controller: _confirmPasswordController,
                      ),
                      const SizedBox(height: 10),

                      // 🔵 Bouton Sign Up
                      MaterialButton(
                        minWidth: double.infinity,
                        height: 55,
                        onPressed: () {
                          if (_validateFields()) {
                            _saveAccount();
                          }
                        },
                        color: const Color(0xff0095FF),
                        elevation: 5,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(50),
                        ),
                        child: const Text(
                          "Sign Up",
                          style: TextStyle(
                            fontWeight: FontWeight.w600,
                            fontSize: 18,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(height: 15),

                      // 🔵 Lien vers Login
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: <Widget>[
                          const Text("avez vous déja un compte"),
                          GestureDetector(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => LoginPage(),
                                ),
                              );
                            },
                            child: const Text(
                              "connexion",
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

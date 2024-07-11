#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>
#include <limits>
#include <fstream>
#include <sstream>
#include <list>
#include "Independency/independency.hpp"

class random
{ public:
  random() { std::srand(std::time(nullptr)); }

  float float_one()
  { float val = (float)std::rand() / (float)RAND_MAX;
    val *= (float)std::rand() / (float)RAND_MAX < 0.5 ? -1 : 1;
    return val; }
} rnd;

class cli_parser
{ public:
  cli_parser(int argc, char** argv)
  { train_file_name = "";
    data_base_file_name = "";
    volume = 4;
    need_training = false;
    
    bool question_key = false;
    bool train_file_name_key = false;
    bool data_base_file_name_key = false;
    bool volume_key = false;
    
    for (int i = 1; i < argc; i++)
    { if (std::string("--input") == argv[i] ||
          std::string("-i") == argv[i])
      { train_file_name_key = true; continue; }

      if (std::string("--db-name") == argv[i] ||
          std::string("-d") == argv[i])
      { data_base_file_name_key = true; continue; }

      if (std::string("--volume") == argv[i] ||
          std::string("-v") == argv[i])
      { volume_key = true; continue; }

      if (std::string("--question") == argv[i] ||
          std::string("-q") == argv[i])
      { question_key = true; continue; }

      if (std::string("--train") == argv[i]) { need_training = true; }

      if (question_key) { question = argv[i]; question_key = false; }

      if (train_file_name_key)
      { train_file_name_key = false;
        train_file_name = argv[i]; }

      if (data_base_file_name_key)
      { data_base_file_name_key = false;
        data_base_file_name = argv[i]; }
    
      if (volume_key)
      { volume_key = false;
        volume = std::stoi(argv[i]);
        if (volume < 0 || volume > 1024) { volume = 4; } } } }

  std::string train_file_name;
  std::string data_base_file_name;
  int volume;
  std::string question;
  bool need_training;
};

#define IV independency::value
#define IM independency::message
#define IC independency::callback
#define IP independency::path
#define root IP()

class trainer
{ public:
  trainer(independency::storage& st,
          independency::storage& db, int vol) : st(st), db(db), vol(vol)
  { if (db.ls(root).size() == 0) { randomize_weights(db); } }

  void print_db()
  { std::printf("\n| Token                |");
    for (unsigned int i = 0; i < vol; i++)
    { std::printf(" idx %3d |", i); }
    for (unsigned int i = 0; i < vol; i++)
    { std::printf(" wgt %3d |", i); }
    std::printf("\n");
    
    std::printf("|----------------------|");
    for (unsigned int i = 0; i < vol; i++)
    { std::printf("---------|---------|"); }
    std::printf("\n");

    for (std::string tok : db.ls(root))
    { std::printf("| %20s |", tok.c_str());
      for (unsigned int i = 0; i < vol; i++)
      { std::printf(" %7.2f |", (float)db[root/tok/"i"/i]); }
      for (unsigned int i = 0; i < vol; i++)
      { std::printf(" %7.2f |", (float)db[root/tok/"w"/i]); }
      std::printf("\n"); } }

  float train(float k = 0.2, int attempts = 1000)
  { float error = dry_run();
    independency::storage tmp_buffer; store(tmp_buffer);
    for (int i = 0; i < attempts; i++)
    { modify(db, k); float err_tmp = dry_run();
      if (err_tmp < error) { return err_tmp; } 
      restore(tmp_buffer); }
    return error; }

  std::list<std::string> test_run(std::string question, int ntokens = 100)
  { std::list<std::string> qtokens = tokenize(question);
    std::list<std::string> tokens;
    tokens.push_back("@q");
    for (std::string tok : qtokens) { tokens.push_back(tok); }
    tokens.push_back("@a");

    float context[vol]; for (int i = 0; i < vol; i++) { context[i] = 0; }

    for (std::string tok : tokens)
    { if (!db.chk(root/tok)) { continue; }
      for (int i = 0; i < vol; i++)
      { context[i] = sigmoid(context[i] + (float)db[root/tok/"w"/i]); } }

    std::list<std::string> atokens;
    std::string candidate = suggest(context);
    int counter = 0;
    while (candidate != "@e" && counter < ntokens)
    { atokens.push_back(candidate);
      for (int i = 0; i < vol; i++)
      { context[i] = sigmoid(context[i] + (float)db[root/candidate/"w"/i]); }
      candidate = suggest(context); counter++; }

    return atokens; }

  void shake(float k = 0.1) { modify(db, k); }

  private:
  float sigmoid(float x) { return 3*x / (1 + abs(3*x)); }
  float abs(float x) { return (x < 0) ? -x : x; }

  void randomize_weights(independency::storage& db)
  { db.parse("");
   
    for (std::string idx : st.ls(root))
    { std::list<std::string> tokens = tokenize(st[root/idx]);
      for (std::string tok : tokens)
      { if (db.chk(root/tok)) { continue; }
        for (int i = 0; i < vol; i++)
        { db[root/tok/"i"/i] = rnd.float_one();
          db[root/tok/"w"/i] = rnd.float_one(); } } } }

  std::string suggest(float* context)
  { std::string res;
    
    float error = std::numeric_limits<float>().max();

    for (std::string tok : db.ls(root))
    { float loc_error = 0;
      for (int i = 0; i < vol; i++)
      { loc_error += abs(context[i] - (float)db[root/tok/"i"/i]); }
      if (loc_error < error) { error = loc_error; res = tok; } }

    return res; }

  float dry_run(bool print = false)
  { float err = 0;

    for (std::string item : st.ls(root))
    { std::list<std::string> tokens = tokenize(st[root/item]);

      if (print)
      { std::printf("\n| N   | Token                |");
        for (int i = 0; i < vol; i++) { std::printf(" ctx %3d |", i); }
        std::printf(" Etotal  | Emoment |\n");

        std::printf(  "|-----|----------------------|");
        for (int i = 0; i < vol + 2; i++) { std::printf("---------|"); }
        std::printf("\n"); }

      float context[vol]; for (int i = 0; i < vol; i++) { context[i] = 0; }
      float context_prev[vol];
      for (int i = 0; i < vol; i++) { context_prev[i] = context[i]; }
      float err_prev = err;

      unsigned int counter = 0;
      for (std::string tok : tokens)
      { for (int i = 0; i < vol; i++)
        { context[i] = sigmoid(context[i] + (float)db[root/tok/"w"/i]); }

        if (counter > 0)
        { float err_tmp = 0;
          for (int i = 0; i < vol; i++)
          { err_tmp += abs((float)db[root/tok/"i"/i] - context_prev[i]); }
          err += err_tmp; }

        if (print)
        { std::printf("| %3d | %20s |", counter, tok.c_str());
          for (int i = 0; i < vol; i++) { std::printf(" %7.2f |", context[i]); }
          std::printf(" %7.2f | %7.2f |\n", err, err - err_prev); }

        for (int i = 0; i < vol; i++) { context_prev[i] = context[i]; }
        err_prev = err;
        counter++; } }

    return err; }

  std::list<std::string> tokenize(std::string in)
  { std::list<std::string> res;

    std::string current;
    for (char c : in)
    { if (c == ' ' || c == '\n' || c == '\t')
      { if (!current.empty()) { res.push_back(current); current.clear(); }
        continue; }

      if (c == '?')
      { res.push_back(current); current.clear(); res.push_back("?"); continue; }

      if (c == '!')
      { res.push_back(current); current.clear(); res.push_back("!"); continue; }

      if (c == '.')
      { res.push_back(current); current.clear(); res.push_back("."); continue; }

      if (c == ',')
      { res.push_back(current); current.clear(); res.push_back(","); continue; }

      current.push_back(c); }

    if (!current.empty()) { res.push_back(current); }

    return res; }

  void store(independency::storage& buffer)   { buffer.parse(db.serialize()); }
  void restore(independency::storage& buffer) { db.parse(buffer.serialize()); }

  void modify(independency::storage& buffer, float k)
  { for (std::string tok : buffer.ls(root))
    { for (int i = 0; i < vol; i++)
      { float i_tmp = buffer[root/tok/"i"/i];
        i_tmp += rnd.float_one() * k;
        if (i_tmp > 1.0f) { i_tmp = 1.0f; }
        else if (i_tmp < -1.0f) { i_tmp = -1.0f; }

        float w_tmp = buffer[root/tok/"w"/i];
        w_tmp += rnd.float_one() * k;
        if (w_tmp > 1.0f) { w_tmp = 1.0f; }
        else if (w_tmp < -1.0f) { w_tmp = -1.0f; }

        buffer[root/tok/"i"/i] = i_tmp;
        buffer[root/tok/"w"/i] = w_tmp; } } }

  int vol;
  independency::storage& st;
  independency::storage& db;
};

int main(int argc, char** argv)
{ std::printf("Small Language Model demo\n");

  cli_parser cp(argc, argv);

  independency::storage training_db;
  if (!cp.train_file_name.empty())
  { std::ifstream file(cp.train_file_name);
    std::stringstream data; data << file.rdbuf(); file.close();
    training_db.parse(data.str()); }

  independency::storage weights_db;
  if (!cp.data_base_file_name.empty())
  { std::ifstream file(cp.data_base_file_name);
    std::stringstream data; data << file.rdbuf(); file.close();
    weights_db.parse(data.str()); }

  trainer t(training_db, weights_db, cp.volume);

  if (cp.need_training)
  { // t.print_db();
    std::printf("\nTraining process\n");
    std::printf("| N   | Error   |\n|-----|---------|\n");
    float error = 0;
    unsigned int counter = 0;
    for (unsigned int i = 0; i < 1000; i++)
    { float error_tmp = t.train(0.02);
      if (i < 800)
      { if (error_tmp == error) { counter++; }
        else { counter = 0; }
        if (counter > 10)
        { t.shake(0.1); error_tmp = t.train(0.02); counter = 0; } }
      error = error_tmp;
      std::printf("| %3d | %7.2f |\r", i, error); std::fflush(stdout); }
    // t.print_db();
  }

  if (!cp.question.empty())
  { std::printf("\nQ: %s\nA: ", cp.question.c_str());
    std::list<std::string> answer = t.test_run(cp.question);
    for (std::string atok : answer) { std::printf("%s ", atok.c_str()); }
    std::printf("\n"); }

  if (!cp.data_base_file_name.empty())
  { std::ofstream file(cp.data_base_file_name);
    file << weights_db.serialize(); file.close(); }

  return 0; }

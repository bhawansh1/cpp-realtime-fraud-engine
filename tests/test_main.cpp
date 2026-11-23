#include <iostream>

void run_matrix_tests();
void run_logistic_tests();
void run_tree_tests();

int main() {
    std::cout << "========================================\n";
    std::cout << "     RUNNING VELOCEML TEST SUITE        \n";
    std::cout << "========================================\n\n";

    try {
        run_matrix_tests();
        run_logistic_tests();
        run_tree_tests();
        std::cout << "\n>>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<<\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST FAILED]: " << e.what() << "\n";
        return 1;
    }
}

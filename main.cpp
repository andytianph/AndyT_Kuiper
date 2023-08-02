#include <iostream>
#include <armadillo>
using  namespace  std;

int test01()
{
    //armadillo是列主序的
    arma::fmat A = "1,2,3;"
                   "4,5,6;"
                   "7,8,9;";
    cout << A << "\n";
    cout << A.at(0) << "\n";
    cout << A.at(1) << "\n";
    cout << A.at(2) << "\n";
    cout << A.at(3) << "\n";
    cout << "-------------------" << endl;
    cout << A.at(0,0) << "\n";
    cout << A.at(0,1) << "\n";
    cout << A.at(1,1) << "\n";

    return 0;
}

int test02()
{
    arma::fmat A = "1,2,3;"
                   "4,5,6;"
                   "7,8,9;";

    arma::fmat X = "1,1,1;"
                   "1,1,1;"
                   "1,1,1;";

    arma::fmat bias = "1,1,1;"
                      "1,1,1;"
                      "1,1,1;";

    arma::fmat output(3, 3);
    //todo 在此处插入代码，完成output = AxX + bias的运算
    output = A * X + bias;
    cout << output << endl;

    return 0;
}

int main() {
  arma::fmat in_1(32, 32, arma::fill::ones);
  arma::fmat in_2(32, 32, arma::fill::ones);

//  std::cout << "*********test**********" << std::endl;
//  test02();
//  std::cout << "*********test_end**********" << std::endl;

  arma::fmat out = in_1 + in_2;
  std::cout << "rows " << out.n_rows << "\n";
  std::cout << "cols " << out.n_cols << "\n";
  std::cout << "value " << out.at(0) << "\n";
  return 0;
}

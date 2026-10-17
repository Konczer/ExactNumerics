#include <boost/multiprecision/cpp_bin_float.hpp>
#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <stdexcept>
namespace py = pybind11;
// Numerical SHO positivity, independent of symbolic determinant construction.
// Return -1 for a negative pivot, 0 for an unresolved pivot, +1 for positive.
template <unsigned Digits>
int sho_numeric_positive(int K,const std::string& energy) {
    using F = boost::multiprecision::number<boost::multiprecision::cpp_bin_float<Digits>>;
    if(K<1) throw std::invalid_argument("Numerical depth must be positive");
    F E(energy);
    std::vector<F> mu(2*size_t(K)-1,F(0)); mu[0]=1;
    for(size_t s=2;s<mu.size();s+=2) {
        mu[s]=F(2)*F(s-1)*E*mu[s-2]/F(s);
        if(s>=4) mu[s]+=F(s-1)*F(s-2)*F(s-3)*mu[s-4]/(F(4)*F(s));
    }
    // Diagonal scaling controls moment growth; LDL still needs high precision.
    const F guard=pow(F(10),-int(Digits)+30);
    for(int p=0;p<2;++p) {
        int n=(K+1-p)/2;
        std::vector<F> scale(n);
        for(int i=0;i<n;++i) {
            if(mu[4*i+2*p]<=0) return -1;
            scale[i]=sqrt(mu[4*i+2*p]);
        }
        std::vector<std::vector<F>> a(n,std::vector<F>(n));
        for(int i=0;i<n;++i) for(int j=0;j<=i;++j)
            a[i][j]=mu[2*(i+j+p)]/(scale[i]*scale[j]);
        for(int k=0;k<n;++k) {
            F pivot=a[k][k];
            if(abs(pivot)<guard) return 0;
            if(pivot<0) return -1;
            for(int i=k+1;i<n;++i) {
                F factor=a[i][k]/pivot;
                for(int j=i;j<n;++j) a[j][i]-=factor*a[j][k];
            }
        }
    }
    return 1;
}
int numeric_positive(int K,const std::string& energy,int precision) {
    if(precision==160) return sho_numeric_positive<160>(K,energy);
    if(precision==240) return sho_numeric_positive<240>(K,energy);
    throw std::invalid_argument("precision must be 160 or 240 decimal digits");
}

PYBIND11_MODULE(_qmb_numeric_v2,m) {
    m.def("sho_numeric_positive", &numeric_positive, py::arg("K"), py::arg("energy"), py::arg("precision")=160,
          py::call_guard<py::gil_scoped_release>());
}

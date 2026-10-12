#include <boost/multiprecision/cpp_int.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

using R = boost::multiprecision::cpp_rational;
using P = std::vector<R>; // coefficients in ascending order
namespace py = pybind11;
void trim(P& p) { while (p.size() > 1 && p.back() == 0) p.pop_back(); }
P mul(const P& a, const P& b) {
    P c(a.size()+b.size()-1, R(0));
    for (size_t i=0;i<a.size();++i)
        for (size_t j=0;j<b.size();++j) c[i+j] += a[i]*b[j];
    trim(c); return c;
}
P sub(P a, const P& b) {
    a.resize(std::max(a.size(),b.size()),R(0));
    for(size_t i=0;i<b.size();++i) a[i]-=b[i];
    trim(a); return a;
}
P divide(P a, const P& b) {
    if (b.size()==1 && b[0]==0) throw std::runtime_error("Zero polynomial divisor");
    P q(a.size()>=b.size()?a.size()-b.size()+1:1,R(0));
    while (!(a.size()==1 && a[0]==0) && a.size()>=b.size()) {
        size_t shift=a.size()-b.size(); R factor=a.back()/b.back();
        q[shift]+=factor;
        for(size_t j=0;j<b.size();++j) a[j+shift]-=factor*b[j];
        trim(a);
    }
    if (!(a.size()==1 && a[0]==0)) throw std::runtime_error("Inexact Bareiss division");
    trim(q); return q;
}
std::vector<P> moments(int K) {
    if(K<1) throw std::invalid_argument("K must be >= 1");
    std::vector<P> mu(2*size_t(K)-1,P{R(0)}); mu[0]={R(1)};
    for(size_t s=2;s<mu.size();s+=2) {
        P p(mu[s-2].size()+1,R(0));
        for(size_t j=0;j<mu[s-2].size();++j) p[j+1]=R(2)*R(s-1)*mu[s-2][j]/R(s);
        if(s>=4) {
            R c=R(s-1)*R(s-2)*R(s-3)/(R(4)*R(s));
            for(size_t j=0;j<mu[s-4].size();++j) p[j]+=c*mu[s-4][j];
        }
        trim(p); mu[s]=std::move(p);
    }
    return mu;
}
P determinant(const std::vector<P>& mu,int K,int parity) {
    int n=(K+1-parity)/2;
    if(n==0) return {R(1)};
    std::vector<std::vector<P>> a(n,std::vector<P>(n));
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) a[i][j]=mu[2*(i+j+parity)];
    P previous{R(1)}; int sign=1;
    for(int k=0;k<n-1;++k) {
        if(a[k][k].size()==1 && a[k][k][0]==0) {
            int row=k+1;
            while(row<n && a[row][k].size()==1 && a[row][k][0]==0) ++row;
            if(row==n) return {R(0)};
            std::swap(a[k],a[row]); sign=-sign;
        }
        P pivot=a[k][k];
        for(int i=k+1;i<n;++i) for(int j=k+1;j<n;++j)
            a[i][j]=divide(sub(mul(pivot,a[i][j]),mul(a[i][k],a[k][j])),previous);
        for(int i=k+1;i<n;++i) a[i][k]={R(0)};
        previous=std::move(pivot);
    }
    P result=a[n-1][n-1]; for(auto& c:result) c*=sign;
    return result;
}
std::vector<std::vector<std::string>> determinants(int K) {
    auto mu=moments(K); std::vector<std::vector<std::string>> out;
    for(int p=0;p<2;++p) {
        std::vector<std::string> coefficients;
        for(const auto& c:determinant(mu,K,p)) coefficients.push_back(c.str());
        out.push_back(std::move(coefficients));
    }
    return out;
}
bool positive_moments(int K, const std::vector<R>& mu) {
    for(int p=0;p<2;++p) {
        int n=(K+1-p)/2;
        std::vector<std::vector<R>> a(n,std::vector<R>(n));
        for(int i=0;i<n;++i) for(int j=0;j<n;++j) a[i][j]=mu[2*(i+j+p)];
        for(int k=0;k<n;++k) {
            R pivot=a[k][k]; if(pivot<=0) return false;
            for(int i=k+1;i<n;++i) for(int j=i;j<n;++j) {
                a[j][i]-=a[i][k]*a[j][k]/pivot;
                a[i][j]=a[j][i];
            }
        }
    }
    return true;
}
bool positive(int K,const std::string& numerator,const std::string& denominator) {
    if(K<1) throw std::invalid_argument("K must be >= 1");
    R E=R(numerator)/R(denominator);
    std::vector<R> mu(2*size_t(K)-1,R(0)); mu[0]=1;
    for(size_t s=2;s<mu.size();s+=2) {
        mu[s]=R(2)*R(s-1)*E*mu[s-2]/R(s);
        if(s>=4) mu[s]+=R(s-1)*R(s-2)*R(s-3)*mu[s-4]/(R(4)*R(s));
    }
    return positive_moments(K,mu);
}
std::vector<R> polynomial_moments(int K, const std::vector<std::string>& coefficients,
                                const std::string& energy, const std::vector<std::string>& seeds,
                                const std::string& mass, const std::string& hbar) {
    if(K<1 || coefficients.size()<3 || coefficients.size()%2==0)
        throw std::invalid_argument("Require K >= 1 and an even degree >= 2");
    int d=static_cast<int>(coefficients.size())-1;
    std::vector<R> v; for(const auto& c:coefficients) v.emplace_back(c);
    if(v.back()<=0) throw std::invalid_argument("Leading coefficient must be positive");
    for(int j=1;j<=d;j+=2) if(v[j]!=0) throw std::invalid_argument("Potential must be even");
    if(seeds.size()!=static_cast<size_t>(d/2-1)) throw std::invalid_argument("Incorrect seed count");
    R E(energy), m(mass), h(hbar);
    if(m<=0 || h<=0) throw std::invalid_argument("mass and hbar must be positive");
    std::vector<R> mu(std::max(2*size_t(K)-1, size_t(d)),R(0)); mu[0]=1;
    for(size_t i=0;i<seeds.size();++i) mu[2*i+2]=R(seeds[i]);
    for(size_t s=d;s<mu.size();s+=2) {
        size_t n=s-d+1;
        R term=R(2)*R(n)*E*mu[n-1];
        if(n>=3) term+=h*h*R(n)*R(n-1)*R(n-2)*mu[n-3]/(R(4)*m);
        for(int j=0;j<d;++j) if(v[j]!=0) term-=(R(2)*R(n)+R(j))*v[j]*mu[n+j-1];
        mu[s]=term/((R(2)*R(n)+R(d))*v[d]);
    }
    mu.resize(2*size_t(K)-1); return mu;
}
std::vector<std::string> polynomial_moment_strings(int K,const std::vector<std::string>& v,
    const std::string& E,const std::vector<std::string>& seeds,const std::string& mass,const std::string& hbar) {
    auto mu=polynomial_moments(K,v,E,seeds,mass,hbar);
    std::vector<std::string> out; for(const auto& x:mu) out.push_back(x.str()); return out;
}
bool polynomial_positive(int K,const std::vector<std::string>& v,const std::string& E,
    const std::vector<std::string>& seeds,const std::string& mass,const std::string& hbar) {
    return positive_moments(K,polynomial_moments(K,v,E,seeds,mass,hbar));
}

PYBIND11_MODULE(_qmb_native,m) {
    m.def("determinant_coefficients",&determinants,py::call_guard<py::gil_scoped_release>());
    m.def("is_positive_definite",&positive,py::call_guard<py::gil_scoped_release>());
    m.def("polynomial_moments",&polynomial_moment_strings,py::call_guard<py::gil_scoped_release>());
    m.def("polynomial_positive",&polynomial_positive,py::call_guard<py::gil_scoped_release>());
}

// Sparse modular row echelon; rank only, with deterministic sorted-vector merging.
// All products are < p^2 and this implementation requires p <= 1,000,000,007.
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
using Pair=std::pair<int,int64_t>;
struct State {
 int n,p,r=0,maxnnz=0; int64_t ops=0; std::vector<int> order; std::vector<std::vector<Pair>> pivot;
 State(int n_,int p_,const int* ord):n(n_),p(p_),order(ord,ord+n_),pivot(n_) {}
 int64_t inverse(int64_t a){int64_t b=p,u=1,v=0;while(b){auto q=a/b;auto t=a-q*b;a=b;b=t;t=u-q*v;u=v;v=t;}u%=p;if(u<0)u+=p;return u;}
 bool add(std::vector<Pair> row){
  if(r==n)return false;
  std::sort(row.begin(),row.end());
  while(!row.empty()){
   int c=row[0].first;int64_t factor=row[0].second;
   if(pivot[c].empty()){
    int64_t inv=inverse(factor);for(auto& z:row)z.second=z.second*inv%p;
    maxnnz=std::max(maxnnz,(int)row.size());pivot[c]=std::move(row);++r;return true;
   }
   auto const& b=pivot[c];std::vector<Pair> out;out.reserve(row.size()+b.size());size_t i=1,j=1;
   while(i<row.size() || j<b.size()){
    int col;int64_t val;
    if(j==b.size() || (i<row.size() && row[i].first<b[j].first)){col=row[i].first;val=row[i].second;++i;}
    else if(i==row.size() || b[j].first<row[i].first){col=b[j].first;val=(-factor*b[j].second)%p;if(val<0)val+=p;++j;}
    else{col=row[i].first;val=(row[i].second-factor*b[j].second)%p;if(val<0)val+=p;++i;++j;}
    if(val)out.emplace_back(col,val);
   }
   ops+=(int64_t)row.size()+(int64_t)b.size();row.swap(out);
  }return false;
 }
};
extern "C" {
void* rank_new(int n,int p,const int* order){return new State(n,p,order);}
void rank_delete(void* handle){delete (State*)handle;}
int rank_value(void* handle){return ((State*)handle)->r;}
int rank_max_nnz(void* handle){return ((State*)handle)->maxnnz;}
int64_t rank_ops(void* handle){return ((State*)handle)->ops;}
void rank_add_csr(void* handle,int nr,const int64_t* ptr,const int32_t* idx,const int64_t* val){
 auto& s=*(State*)handle;
 for(int i=0;i<nr && s.r<s.n;++i){
  std::vector<Pair> row;row.reserve(ptr[i+1]-ptr[i]);
  for(int64_t q=ptr[i];q<ptr[i+1];++q){int64_t z=val[q]%s.p;if(z<0)z+=s.p;if(z)row.emplace_back(s.order[idx[q]],z);}
  s.add(std::move(row));
 }
}
}

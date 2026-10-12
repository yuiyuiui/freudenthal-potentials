// Exact verification on unbounded integer parameter chambers.
// No floating point arithmetic, random tests, or rank oracle is used here.
#include <algorithm>
#include <array>
#include <cassert>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using I=long long;
using V=std::array<int,3>;
using Pos=std::array<int,4>;
const int PERM[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
std::string context;
void demand(bool b,const std::string&s){if(!b)throw std::runtime_error(context+": "+s);}
struct Form {std::array<I,4> c{};I b=0;};
Form operator+(Form a,const Form&b){for(int i=0;i<4;i++)a.c[i]+=b.c[i];a.b+=b.b;return a;}
Form operator-(Form a,const Form&b){for(int i=0;i<4;i++)a.c[i]-=b.c[i];a.b-=b.b;return a;}
Form operator*(Form a,I m){for(auto&x:a.c)x*=m;a.b*=m;return a;}
Form constant(I b){Form f;f.b=b;return f;}
struct Region {int m=4;std::array<int,4> lo{},rep{};std::array<bool,4> var{};int sum_lower=7;
 bool initialize(){rep=lo;int sum=0,first=-1;for(int i=0;i<m;i++){sum+=lo[i];if(var[i]&&first<0)first=i;}if(sum<sum_lower){if(first<0)return false;rep[first]+=sum_lower-sum;}return true;}
 I eval(const Form&f)const{I v=f.b;for(int i=0;i<m;i++)v+=f.c[i]*rep[i];return v;}
 bool ge(const Form&f,I bound=0)const{I v=f.b;int sum=0;I smallest=1000000000;for(int i=0;i<m;i++){sum+=lo[i];v+=f.c[i]*lo[i];if(var[i]){if(f.c[i]<0)return false;smallest=std::min(smallest,f.c[i]);}}if(sum<sum_lower){if(smallest==1000000000)return true;v+=(sum_lower-sum)*smallest;}return v>=bound;}
 bool eq(const Form&f,I value=0)const{for(int i=0;i<m;i++)if(var[i]&&f.c[i]!=0)return false;return eval(f)==value;}
};
int encode(const V&bc,const V&pm,const std::array<int,4>&g,int co){int p=-1;for(int j=0;j<6;j++)if(std::equal(pm.begin(),pm.end(),PERM[j]))p=j;demand(p>=0,"bad permutation");int b=bc[0]*9+bc[1]*3+bc[2],gg=g[0]*64+g[1]*16+g[2]*4+g[3];return ((b*6+p)*256+gg)*3+co;}
struct Term {V delta;int co;I value;};
struct Rule {int kind=-1,id=-1,co=0,mask=0;V bc{},pm{};std::array<int,4> gap{};std::vector<Term>row;};
std::vector<Rule>D(27*6*256*3);std::vector<int>ids;std::vector<int>owners(D.size());
int entity_mask(const V&pm,const std::array<int,4>&g){V u{};int mask=0;for(int i=0;i<4;i++){if(g[i]>0)mask|=1<<(u[0]*4+u[1]*2+u[2]);if(i<3)u[pm[i]]=1;}return mask;}
struct State {V cell{},bc{},pm{};std::array<int,4>gap{};int key(int co)const{return encode(bc,pm,gap,co);}};
State classify(const std::array<Form,3>&q,const Form&degree,int N,const Region&r){
 State s;I d=r.eval(degree);demand(d>=7,"degree below target");std::array<Form,3> rem;std::array<I,3>rv;
 for(int j=0;j<3;j++){
  demand(r.ge(q[j])&&r.ge(degree*N-q[j]),"coefficient leaves physical domain");I x=r.eval(q[j]);s.cell[j]=std::min<I>(x/d,N-1);demand(s.cell[j]>=0&&s.cell[j]<N,"invalid cell");
  rem[j]=q[j]-degree*s.cell[j];rv[j]=r.eval(rem[j]);
  demand(r.ge(rem[j]),"cell lower inequality not uniform");
  demand(r.ge(degree-rem[j],s.cell[j]<N-1?1:0),"cell upper inequality not uniform");
  s.bc[j]=(s.cell[j]==0?0:s.cell[j]==N-1?2:1);
 }
 s.pm={0,1,2};std::sort(s.pm.begin(),s.pm.end(),[&](int i,int j){return rv[i]!=rv[j]?rv[i]>rv[j]:i<j;});
 for(int t=0;t<2;t++)demand(r.ge(rem[s.pm[t]]-rem[s.pm[t+1]],s.pm[t]>s.pm[t+1]?1:0),"order not uniform");
 std::array<Form,4>g={degree-rem[s.pm[0]],rem[s.pm[0]]-rem[s.pm[1]],rem[s.pm[1]]-rem[s.pm[2]],rem[s.pm[2]]};
 for(int j=0;j<4;j++){I v=r.eval(g[j]);demand(v>=0,"negative barycentric coordinate");s.gap[j]=std::min<I>(v,3);demand(v<3?r.eq(g[j],v):r.ge(g[j],3),"clipped gap not uniform");}
 return s;
}
void load(const std::string&path){std::ifstream f(path);demand(bool(f),"rules file missing");int nr;f>>nr;for(int i=0;i<nr;i++){Rule r;int nn;f>>r.id>>r.kind;for(auto&x:r.bc)f>>x;for(auto&x:r.pm)f>>x;for(auto&x:r.gap)f>>x;f>>r.co>>r.mask>>nn;demand(nn>=0&&nn<=1000,"row-length input bound");for(int j=0;j<nn;j++){Term t;for(auto&x:t.delta)f>>x;f>>t.co>>t.value;demand(t.co>=0&&t.co<3&&std::abs(t.value)<=1024,"coefficient input bound");for(int z:t.delta)demand(std::abs(z)<=7,"offset input bound");r.row.push_back(t);}demand(bool(f),"truncated rule");demand(r.id==encode(r.bc,r.pm,r.gap,r.co),"rule id mismatch");demand(D[r.id].kind==-1,"duplicate rule");demand(r.mask==entity_mask(r.pm,r.gap),"entity mask mismatch");D[r.id]=r;ids.push_back(r.id);owners[r.id]=r.mask;}demand(nr==59943,"unexpected rule count");}
struct Location{int N;V a;};
I pivot_regions=0,pivot_contexts=0,pivot_links=0,owner_updates=0;
void check_pivots(){
 std::array<std::vector<Location>,27>loc;
 for(int N:{2,3,5})for(int a=0;a<N;a++)for(int b=0;b<N;b++)for(int c=0;c<N;c++){V v={a,b,c},bc;for(int j=0;j<3;j++)bc[j]=v[j]==0?0:v[j]==N-1?2:1;loc[bc[0]*9+bc[1]*3+bc[2]].push_back({N,v});}
 int done=0;
 for(int id:ids){const Rule&p=D[id];if(p.kind!=1)continue;context="pivot rule "+std::to_string(id);int rad=0,ndiag=0;for(auto&t:p.row){for(int x:t.delta)rad=std::max(rad,std::abs(x));if(t.delta==V{0,0,0}&&t.co==p.co){demand(t.value==1,"non-unit pivot");ndiag++;}demand(t.value!=0,"explicit zero entry");}demand(ndiag==1&&rad<=7,"bad diagonal/radius");int M=std::max(7,2*rad+3);
 Region r;r.m=4;r.sum_lower=7;
 std::function<void(int)>go=[&](int j){
  if(j<4){if(p.gap[j]<3){r.lo[j]=p.gap[j];r.var[j]=false;go(j+1);}else for(int v=3;v<=M;v++){r.lo[j]=v;r.var[j]=(v==M);go(j+1);}return;}
  if(!r.initialize()) return;
  pivot_regions++;
  Form deg;for(auto&x:deg.c)x=1;
  std::array<Form,3>R;for(int i=0;i<3;i++)for(int z=i+1;z<4;z++)R[p.pm[i]].c[z]=1;
  for(auto&l:loc[p.bc[0]*9+p.bc[1]*3+p.bc[2]]){
   pivot_contexts++;std::array<Form,3>q;for(int ax=0;ax<3;ax++)q[ax]=deg*l.a[ax]+R[ax];
   auto initial=classify(q,deg,l.N,r);demand(initial.key(p.co)==id,"anchor state mismatch");
   for(auto&t:p.row){if(t.delta==V{0,0,0}&&t.co==p.co)continue;pivot_links++;
    auto q1=q;for(int ax=0;ax<3;ax++)q1[ax].b+=t.delta[ax];auto s=classify(q1,deg,l.N,r);int f=s.key(t.co);demand(D[f].kind==0,"off-diagonal pivot or absent rule");
    int allowed=0;for(int v=0;v<8;v++)if(p.mask&(1<<v)){V u={v/4,(v/2)%2,v%2};bool valid=true;int idx=0;for(int ax=0;ax<3;ax++){int z=l.a[ax]+u[ax]-s.cell[ax];if(z<0||z>1){valid=false;break;}idx=2*idx+z;}if(valid)allowed|=1<<idx;}
    int old=owners[f];owners[f]&=allowed;demand(owners[f]!=0,"empty universal vertex-owner intersection");if(old!=owners[f])owner_updates++;
   }
  }
 };
 go(0);done++;if(done%3000==0)std::cerr<<"pivots "<<done<<" regions "<<pivot_regions<<" links "<<pivot_links<<"\n";
 }
 for(int id:ids)if(D[id].kind==0)demand(owners[id]!=0,"unassigned free owner");
}
using Raw=std::map<Pos,I>;
void add(Raw&r,Pos p,I z){I v=r[p]+z;if(v)r[p]=v;else r.erase(p);}
struct Face{int N;V origin;std::array<V,3>fv;std::array<std::array<V,4>,2>tv;};
std::vector<Face>load_faces(const std::string&path){std::ifstream f(path);demand(bool(f),"faces file missing");int n;f>>n;std::vector<Face>out;for(int i=0;i<n;i++){Face a;f>>a.N;for(auto&x:a.origin)f>>x;for(auto&v:a.fv)for(auto&x:v)f>>x;for(auto&t:a.tv)for(auto&v:t)for(auto&x:v)f>>x;demand(bool(f),"truncated face");out.push_back(a);}return out;}
std::array<V,4> gradients(const std::array<V,4>&t){std::array<V,4>g{};std::array<int,3>p{};for(int j=0;j<3;j++){int axis=-1;for(int k=0;k<3;k++){int z=t[j+1][k]-t[j][k];demand(z==0||z==1,"non-Freudenthal tetrahedron");if(z){demand(axis==-1,"multiple step axes");axis=k;}}demand(axis>=0,"zero step");p[j]=axis;}demand(p[0]!=p[1]&&p[0]!=p[2]&&p[1]!=p[2],"repeated step");g[0][p[0]]=-1;g[1][p[0]]=1;g[1][p[1]]=-1;g[2][p[1]]=1;g[2][p[2]]=-1;g[3][p[2]]=1;return g;}
I face_regions=0,face_rows=0,face_points=0;
void check_faces(const std::vector<Face>&faces,int B){
 int fi=0;
 for(auto&f:faces){context="face profile "+std::to_string(fi);std::array<Raw,3>rr;
  for(int t=0;t<2;t++){auto g=gradients(f.tv[t]);int sign=t==0?1:-1;for(int j=0;j<4;j++){auto z=g[j];I cross[3][3]={{0,-z[2],z[1]},{z[2],0,-z[0]},{-z[1],z[0],0}};for(int co=0;co<3;co++)for(int c=0;c<3;c++)if(cross[co][c])add(rr[co],{f.tv[t][j][0],f.tv[t][j][1],f.tv[t][j][2],c},sign*cross[co][c]);}}
  for(int b0=0;b0<=B;b0++)for(int b1=0;b1<=B;b1++)for(int b2=0;b2<=B;b2++){
   Region r;r.m=3;r.lo={b0,b1,b2,0};r.var={b0==B,b1==B,b2==B,false};r.sum_lower=6;if(!r.initialize())continue;face_regions++;
   Form degree;degree.c={1,1,1,0};degree.b=1;
   for(int comp=0;comp<3;comp++){
    if(rr[comp].empty()) continue;
    face_rows++;Raw residue=rr[comp];
    for(auto&[at,z]:rr[comp]){
     face_points++;std::array<Form,3>q;for(int ax=0;ax<3;ax++){for(int j=0;j<3;j++)q[ax].c[j]=f.origin[ax]+f.fv[j][ax];q[ax].b=f.origin[ax]+at[ax];}
     auto st=classify(q,degree,f.N,r);int id=st.key(at[3]);demand(D[id].kind>=0,"missing face coefficient rule");if(D[id].kind==0)continue;
     for(auto&t:D[id].row)add(residue,{at[0]+t.delta[0],at[1]+t.delta[1],at[2]+t.delta[2],t.co},-z*t.value);
    }
    demand(residue.empty(),"reverse row-space identity fails");
   }
  }
  fi++;if(fi%100==0)std::cerr<<"faces "<<fi<<" regions "<<face_regions<<" points "<<face_points<<"\n";
 }
}
int main(int argc,char**argv){
 try{demand(argc>=3,"usage: exact_parameter_check results_dir mode [beta_cap]");std::string root=argv[1],mode=argv[2];load(root+"/rules_integer.txt");
  if(mode=="pivots"||mode=="all"){
   check_pivots();std::ofstream own(root+"/universal_owners.txt");int nf=0;for(int id:ids)if(D[id].kind==0)nf++;own<<nf<<"\n";for(int id:ids)if(D[id].kind==0)own<<id<<" "<<owners[id]<<"\n";
   std::ofstream o(root+"/pivot_and_support_verified.json");o<<"{\"passed\":true,\"unbounded_parameter_chambers\":"<<pivot_regions<<",\"boundary_context_chambers\":"<<pivot_contexts<<",\"off_pivot_links\":"<<pivot_links<<",\"owner_intersections_changed\":"<<owner_updates<<",\"free_rule_owners\":"<<nf<<",\"arithmetic\":\"exact signed integers and affine cone inequalities\"}\n";std::cout<<"pivot and support checks passed\n";
  }
  if(mode=="faces"||mode=="all"){
   int B=argc>3?std::stoi(argv[3]):5;auto faces=load_faces(root+"/face_profiles.txt");check_faces(faces,B);
   std::ofstream o(root+"/reverse_identity_verified.json");o<<"{\"passed\":true,\"face_profiles\":"<<faces.size()<<",\"beta_cap\":"<<B<<",\"unbounded_parameter_chambers\":"<<face_regions<<",\"nonzero_row_identities\":"<<face_rows<<",\"coefficient_classifications\":"<<face_points<<",\"arithmetic\":\"exact signed integers and affine cone inequalities\"}\n";std::cout<<"reverse identity checks passed\n";
  }
  return 0;
 }catch(const std::exception&e){std::cerr<<"FAILED: "<<e.what()<<"\n";return 1;}
}

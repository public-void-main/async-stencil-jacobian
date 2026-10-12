// Build: g++ -O3 -ffp-contract=off -std=c++17 async_emulation.cpp -o emu
// Run:   ./emu 64 gaussian 10 4 0      (N source r D schedule[0=lockstep,1=seq])
//        Keep N small (64-128): N=512 needs ~500k iterations and takes very long on CPU.
// CPU emulation of async_kernel.cuh logic (32x8 tiles). Two block schedules:
//  LOCKSTEP : all blocks do round j together; halos come from a snapshot at round start (block-Jacobi)
//  SEQ      : blocks run one after another, each doing all its rounds of the launch (extreme staleness for waves)
#define NO_MAIN
#include "common.h"
#include "cpu_reference.cpp"
#include <cstdlib>
const int TXe=32, TYe=8;
struct Blk { float s[2][TYe+2][TXe+2]; };
// one block, one round: nsw local sweeps, halo frozen. src = global used for halo (+own tile load); dst = write-back
static void block_round(int bx,int by,const float*src,float*dst,const Problem&p,int nsw,bool last_round,bool want_res,double&mx,bool first,Blk&b,int&cur){
  int N=p.N;
  if(first){ for(int ty=0;ty<TYe;ty++)for(int tx=0;tx<TXe;tx++) b.s[0][ty+1][tx+1]=src[(size_t)(by*TYe+ty)*N+bx*TXe+tx]; cur=0; }
  for(int ty=0;ty<TYe;ty++)for(int tx=0;tx<TXe;tx++){ int gx=bx*TXe+tx, gy=by*TYe+ty;
    if(tx==0){float v=gx>0?src[(size_t)gy*N+gx-1]:0; b.s[0][ty+1][0]=v;b.s[1][ty+1][0]=v;}
    if(tx==TXe-1){float v=gx<N-1?src[(size_t)gy*N+gx+1]:0; b.s[0][ty+1][TXe+1]=v;b.s[1][ty+1][TXe+1]=v;}
    if(ty==0){float v=gy>0?src[(size_t)(gy-1)*N+gx]:0; b.s[0][0][tx+1]=v;b.s[1][0][tx+1]=v;}
    if(ty==TYe-1){float v=gy<N-1?src[(size_t)(gy+1)*N+gx]:0; b.s[0][TYe+1][tx+1]=v;b.s[1][TYe+1][tx+1]=v;} }
  for(int k=0;k<nsw;k++){
    for(int ty=0;ty<TYe;ty++)for(int tx=0;tx<TXe;tx++){ int gx=bx*TXe+tx, gy=by*TYe+ty; bool fx=gx==0||gy==0||gx==N-1||gy==N-1;
      float old=b.s[cur][ty+1][tx+1], v=old;
      if(!fx){ float sum=((b.s[cur][ty][tx+1]+b.s[cur][ty+2][tx+1])+b.s[cur][ty+1][tx])+b.s[cur][ty+1][tx+2];
        v=((old+p.r*sum)+p.src[(size_t)gy*N+gx])/p.denom; if(want_res&&last_round&&k==nsw-1) mx=std::max(mx,(double)std::fabs(v-old)); }
      b.s[cur^1][ty+1][tx+1]=v; }
    cur^=1; }
  for(int ty=0;ty<TYe;ty++)for(int tx=0;tx<TXe;tx++){ int gx=bx*TXe+tx, gy=by*TYe+ty; bool fx=gx==0||gy==0||gx==N-1||gy==N-1;
    if(!fx) dst[(size_t)gy*N+gx]=b.s[cur][ty+1][tx+1]; }
}
int main(int argc,char**argv){
  Config cfg; cfg.N=atoi(argv[1]); parse_source(argv[2],cfg.source); cfg.r=atof(argv[3]); int D=atoi(argv[4]); int sched=atoi(argv[5]); // 0 lockstep 1 seq
  Problem p=init_problem(cfg); int N=cfg.N, BX=N/TXe, BY=N/TYe; int M=10;
  std::vector<float> g=p.u0, snap; std::vector<Blk> blks((size_t)BX*BY); std::vector<int> curs((size_t)BX*BY);
  long long it=0; double res=0; bool conv=false;
  while(it<cfg.max_iters){
    double mx=0; int rounds=(M+D-1)/D;
    if(sched==0){
      for(int rd=0;rd<rounds;rd++){ int nsw=std::min(D,M-rd*D); snap=g;
        for(int by=0;by<BY;by++)for(int bx=0;bx<BX;bx++) block_round(bx,by,snap.data(),g.data(),p,nsw,rd==rounds-1,true,mx,rd==0,blks[by*BX+bx],curs[by*BX+bx]); }
    } else {
      for(int by=0;by<BY;by++)for(int bx=0;bx<BX;bx++) for(int rd=0;rd<rounds;rd++){ int nsw=std::min(D,M-rd*D);
        block_round(bx,by,g.data(),g.data(),p,nsw,rd==rounds-1,true,mx,rd==0,blks[by*BX+bx],curs[by*BX+bx]); }
    }
    it+=M; res=mx; if(res<cfg.tol){conv=true;break;}
  }
  Result ref=cpu_solve(cfg,p,0);
  printf("N=%d src=%s r=%g D=%d sched=%s : conv=%d iters=%lld (sync %lld, R_I=%.2f) maxdiff_vs_sync_grid=%.3e\n",N,argv[2],cfg.r,D,sched?"SEQ":"LOCKSTEP",(int)conv,it,ref.iterations,(double)it/ref.iterations,max_abs_diff(g,ref.final_grid));
}

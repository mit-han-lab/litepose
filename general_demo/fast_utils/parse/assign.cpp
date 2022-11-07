#include "assign.hpp"
#include<cmath>
#include<cstdio>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

namespace trt_pose {
namespace parse {

float dist(float x, float y){
  return sqrt((x - y) * (x - y));
}

bool match(int *mat, const float G[10][10], bool *S, bool *T, float *Lx, float *Ly, float *slack, const int n, int u){
  int i;
  S[u] = true;
  for (i = 0; i < n; i++){
    if (T[i]) continue;
    float t = Lx[u] + Ly[i] - G[u][i];
    if (abs(t) < 1e-2){
      T[i] = true;
      if (mat[i] == -1 || match(mat, G, S, T, Lx, Ly, slack, n, mat[i])){
        mat[i] = u;
        return true;
      }
    }else slack[i] = MIN(slack[i], t);
  }
  return false;
}

void update(bool *S, bool *T, float *Lx, float *Ly, float *slack, const int n){
  int i;
  float d = 1e8;
  for (i = 0; i < n; i++)
    d = MIN(d, slack[i]);
  for (i = 0; i < n; i++){
    if (S[i]) Lx[i] -= d;
    if (T[i]) Ly[i] += d;
  }
}

void KM(int *ch, const float G[10][10], const int n){
  int i, j, k, mat[10];
  float Lx[10], Ly[10], slack[10];
  bool S[10], T[10];
  for (i = 0; i < n; i++){
    Lx[i] = -1e6;
    Ly[i] = 0;
    mat[i] = -1;
    for (j = 0; j < n; j++)
      Lx[i] = MAX(Lx[i], G[i][j]);
  }
  for (i = 0; i < n; i++){
    for (j = 0; j < n; j++) slack[j] = 1e6;
    while(true) {
      for (j = 0; j < n; j++) S[j] = T[j] = false;
      if(match(mat, G, S, T, Lx, Ly, slack, n, i)) break;
      else update(S, T, Lx, Ly, slack, n);
    }
  }
  for (i = 0; i < n; i++) ch[mat[i]] = i;
}

void assign_out(int *num_person,            // 1
                float *ans,                 // MxCx4
                const int *cnt,             // C
                const float *val,           // CxM
                const float *tag,           // CxM
                const int *ind,             // CxMx2
                const int *joint_order,     // 17
                const int C, const int M, 
                const float threshold) {
  int id, i, j, k, num = 0, nj[10], ch[10];
  float G[10][10], diff[10][10], sum[10];
  for (id = 0; id < C; id++){
    i = joint_order[id];
    if (cnt[i] == 0) continue;
    if (num == 0){
      num = cnt[i];
      for (j = 0; j < num; j++){
        int p = i * M + j, q = (j * C + i) << 2;
        ans[q] = ind[p << 1];
        ans[q | 1] = ind[(p << 1) | 1];
        ans[q | 2] = val[p];
        ans[q | 3] = tag[p];
        nj[j] = 1, sum[j] = tag[p];
      }
      continue;
    }
    int num_add = MAX(num, cnt[i]);
    for (j = 0; j < num_add; j++)
      for (k = 0; k < num_add; k++){
        int pre = i * M + k;
        diff[j][k] = (j >= num || k >= cnt[i]) ? 1e4 : dist(1.0 * sum[j] / nj[j], tag[pre]);
        G[j][k] = (j >= num || k >= cnt[i]) ? -1e4 : - (dist(1.0 * sum[j] / nj[j], tag[pre]) * 100 - val[pre]);
      }
    KM(ch, G, num_add);
    int old_num = num;
    for (j = 0; j < num_add; j++){
      if (ch[j] >= cnt[i]) continue;
      if (j < old_num && ch[j] < cnt[i] && diff[j][ch[j]] < threshold){
        int p = i * M + ch[j], q = (j * C + i) << 2;
        ans[q] = ind[p << 1];
        ans[q | 1] = ind[(p << 1) | 1];
        ans[q | 2] = val[p];
        ans[q | 3] = tag[p];
        nj[j]++, sum[j] += tag[p];
      }else{
        if (num == M) continue;
        int p = i * M + ch[j], q = (num * C + i) << 2;
        ans[q] = ind[p << 1];
        ans[q | 1] = ind[(p << 1) | 1];
        ans[q | 2] = val[p];
        ans[q | 3] = tag[p];
        nj[num] = 1, sum[num] = tag[p];
        num ++;
      }
    }
  }
  *num_person = num;
}

} // namespace parse
} // namespace trt_pose

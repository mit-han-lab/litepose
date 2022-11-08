#pragma once

namespace trt_pose {
namespace parse {


bool match(int *ch, bool *S, bool *T, float *Lx, float *Ly, float *slack, const int n, int u);

void update(bool *S, bool *T, float Lx, float *Ly, float *slack, const int n);

void KM(int *ch, const float G[10][10], const int n);

void assign_out(int *num_person,            // 1
                float *ans,                 // MxCx4
                const int *cnt,             // C
                const float *val,           // CxM
                const float *tag,           // CxM
                const int *ind,             // CxMx2
                const int *joint_order,     // 17
                const int C, const int M, 
                const float threshold);

} // namespace parse
} // namespace trt_pose

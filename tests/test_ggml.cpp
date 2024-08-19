#include <stdio.h>
#include <stdlib.h>
#include "ggml.h"

int main() {
    struct ggml_init_params params = {
        .mem_size   = 16*1024*1024,
        .mem_buffer = NULL,
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_param(ctx, x);
    
    struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    
    // Set parameter values
    ggml_set_f32(a, 2.0f);  // a = 2
    ggml_set_f32(b, 3.0f);  // b = 3

    struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
    struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);

    // Create computation plan
    struct ggml_cplan plan = ggml_graph_plan(gf, GGML_DEFAULT_N_THREADS);

    for (int i = 0; i < 10; i++) {
        // Set the value of input x
        ggml_set_f32(x, (float)i);

        // Execute computation
        ggml_graph_compute(gf, &plan);

        // Get and print the result
        float result = ggml_get_f32_1d(f, 0);
        printf("f(%d) = %f\n", i, result);
    }
    
    ggml_free(ctx);

    return 0;
}
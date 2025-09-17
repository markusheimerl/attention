#include "attention.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Test parameters
    int seq_len = 4;
    int d_model = 8;
    int batch_size = 2;
    bool is_causal = false;
    
    printf("Testing backward buffer sharing...\n");
    
    // Create first attention layer (will own its buffers)
    Attention* attn1 = init_attention(seq_len, d_model, batch_size, is_causal, NULL);
    printf("Created attn1 with owns_grad_buffers = %s\n", attn1->owns_grad_buffers ? "true" : "false");
    printf("attn1 grad_output pointer: %p\n", (void*)attn1->grad_output);
    
    // Create second attention layer sharing buffers from first
    Attention* attn2 = init_attention(seq_len, d_model, batch_size, is_causal, attn1);
    printf("Created attn2 with owns_grad_buffers = %s\n", attn2->owns_grad_buffers ? "true" : "false");
    printf("attn2 grad_output pointer: %p\n", (void*)attn2->grad_output);
    
    // Verify buffer sharing
    bool buffers_shared = (attn1->grad_output == attn2->grad_output &&
                          attn1->grad_attn_output == attn2->grad_attn_output &&
                          attn1->grad_weights == attn2->grad_weights &&
                          attn1->grad_scores == attn2->grad_scores &&
                          attn1->grad_Q == attn2->grad_Q &&
                          attn1->grad_K == attn2->grad_K &&
                          attn1->grad_V == attn2->grad_V);
    
    printf("Backward buffers shared: %s\n", buffers_shared ? "YES" : "NO");
    
    // Test that both have different forward buffers (should NOT be shared)
    bool forward_buffers_different = (attn1->Q != attn2->Q &&
                                     attn1->K != attn2->K &&
                                     attn1->V != attn2->V &&
                                     attn1->scores != attn2->scores &&
                                     attn1->attn_weights != attn2->attn_weights &&
                                     attn1->attn_output != attn2->attn_output &&
                                     attn1->output != attn2->output);
    
    printf("Forward buffers separate: %s\n", forward_buffers_different ? "YES" : "NO");
    
    // Create third attention layer without predecessor (should own its buffers)
    Attention* attn3 = init_attention(seq_len, d_model, batch_size, is_causal, NULL);
    printf("Created attn3 with owns_grad_buffers = %s\n", attn3->owns_grad_buffers ? "true" : "false");
    printf("attn3 grad_output pointer: %p\n", (void*)attn3->grad_output);
    
    bool attn3_different = (attn3->grad_output != attn1->grad_output);
    printf("attn3 has different buffers: %s\n", attn3_different ? "YES" : "NO");
    
    // Free in reverse order (should work correctly)
    free_attention(attn3);
    printf("Freed attn3 successfully\n");
    
    free_attention(attn2);
    printf("Freed attn2 successfully (buffers not freed since not owned)\n");
    
    free_attention(attn1);
    printf("Freed attn1 successfully (buffers freed since owned)\n");
    
    printf("Test completed successfully!\n");
    return 0;
}
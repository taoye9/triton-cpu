tt.func public @rewrite_load(%arg0: !tt.ptr<f16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c128_i64 = arith.constant 128 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
  %0 = tt.make_tensor_ptr %arg0, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %load = tt.load %0 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
  tt.return
}
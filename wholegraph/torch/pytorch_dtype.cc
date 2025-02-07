/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "pytorch_dtype.h"

namespace whole_graph {

namespace pytorch {

whole_graph::WMType C10ScalarToWMType(c10::ScalarType st) {
  switch (st) {
    case c10::ScalarType::Byte: return whole_graph::WMT_Uint8;
    case c10::ScalarType::Char: return whole_graph::WMT_Int8;
    case c10::ScalarType::Short: return whole_graph::WMT_Int16;
    case c10::ScalarType::Int: return whole_graph::WMT_Int32;
    case c10::ScalarType::Long: return whole_graph::WMT_Int64;
    case c10::ScalarType::Half: return whole_graph::WMT_Half;
    case c10::ScalarType::Float: return whole_graph::WMT_Float;
    case c10::ScalarType::Double: return whole_graph::WMT_Double;
    case c10::ScalarType::BFloat16: return whole_graph::WMT_Bfloat16;
    default:
      std::cerr << "Scalar type " << st << " not supported.\n";
      abort();
      return whole_graph::WMT_Count;
  }
}

c10::ScalarType WMTypeToC10Scalar(whole_graph::WMType wmt) {
  switch (wmt) {
    case whole_graph::WMT_Uint8: return c10::ScalarType::Byte;
    case whole_graph::WMT_Int8: return c10::ScalarType::Char;
    case whole_graph::WMT_Int16: return c10::ScalarType::Short;
    case whole_graph::WMT_Int32: return c10::ScalarType::Int;
    case whole_graph::WMT_Int64: return c10::ScalarType::Long;
    case whole_graph::WMT_Half: return c10::ScalarType::Half;
    case whole_graph::WMT_Float: return c10::ScalarType::Float;
    case whole_graph::WMT_Double: return c10::ScalarType::Double;
    case whole_graph::WMT_Bfloat16: return c10::ScalarType::BFloat16;
    default:
      std::cerr << "Scalar type " << wmt << " not supported.\n";
      abort();
      return c10::ScalarType::Undefined;
  }
}

}// namespace pytorch

}// namespace whole_graph
#ifndef STATSWRITER_H
#define STATSWRITER_H

#include <CL/cl.h>

class StatsWriter
{
    public:
        static void Write(Context* context, string fileName, char sep)
        {
            cout << "Writing device info file to " << fileName << " ... ";

            ofstream os(fileName);

            os << "CL_DEVICE_ADDRESS_BITS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_ADDRESS_BITS) << endl;
            os << "CL_DEVICE_AVAILABLE" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_AVAILABLE) << endl;
#ifdef CL_VERSION_1_2
			os << "CL_DEVICE_BUILT_IN_KERNELS" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_BUILT_IN_KERNELS) << endl;
#endif
            os << "CL_DEVICE_COMPILER_AVAILABLE" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_AVAILABLE) << endl;

            {
                os << "CL_DEVICE_DOUBLE_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfoWithDefaultOnError<cl_device_fp_config>(CL_DEVICE_DOUBLE_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
                if(flags & CL_FP_FMA)
                    os << "CL_FP_FMA ";
				if(flags & CL_FP_SOFT_FLOAT)
					os << "CL_FP_SOFT_FLOAT ";
                os << endl;
            }

            os << "CL_DEVICE_ENDIAN_LITTLE" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_ENDIAN_LITTLE) << endl;
            os << "CL_DEVICE_ERROR_CORRECTION_SUPPORT" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_ERROR_CORRECTION_SUPPORT) << endl;

            {
                os << "CL_DEVICE_EXECUTION_CAPABILITIES" << sep;
                cl_device_exec_capabilities flags = context->getInfoWithDefaultOnError<cl_device_exec_capabilities>(CL_DEVICE_EXECUTION_CAPABILITIES);
                if(flags & CL_EXEC_KERNEL)
                    os << "CL_EXEC_KERNEL ";
                if(flags & CL_EXEC_NATIVE_KERNEL)
                    os << "CL_EXEC_NATIVE_KERNEL ";
                os << endl;
            }

            os << "CL_DEVICE_EXTENSIONS" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_EXTENSIONS) << endl;
            os << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE" << sep << context->getInfoWithDefaultOnError<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) << endl;

            {
                os << "CL_DEVICE_GLOBAL_MEM_CACHE_TYPE" << sep;
                cl_device_mem_cache_type cacheType = context->getInfoWithDefaultOnError<cl_device_mem_cache_type>(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
                switch(cacheType)
                {
                    case CL_NONE:
                        os << "CL_NONE";
                        break;
                    case CL_READ_ONLY_CACHE:
                        os << "CL_READ_ONLY_CACHE";
                        break;
                    case CL_READ_WRITE_CACHE:
                        os << "CL_READ_WRITE_CACHE";
                        break;
                }
                os << endl;
            }

            os << "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << endl;
            os << "CL_DEVICE_GLOBAL_MEM_SIZE" << sep << context->getInfoWithDefaultOnError<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) << endl;

            {
                os << "CL_DEVICE_HALF_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfoWithDefaultOnError<cl_device_fp_config>(CL_DEVICE_HALF_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
                if(flags & CL_FP_FMA)
                    os << "CL_FP_FMA ";
                if(flags & CL_FP_SOFT_FLOAT)
                    os << "CL_FP_SOFT_FLOAT ";
                os << endl;
            }

            os << "CL_DEVICE_HOST_UNIFIED_MEMORY" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_HOST_UNIFIED_MEMORY) << endl;
            os << "CL_DEVICE_IMAGE_SUPPORT" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_IMAGE_SUPPORT) << endl;
            os << "CL_DEVICE_IMAGE2D_MAX_HEIGHT" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE2D_MAX_HEIGHT) << endl;
            os << "CL_DEVICE_IMAGE2D_MAX_WIDTH" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE2D_MAX_WIDTH) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_DEPTH" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE3D_MAX_DEPTH) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_HEIGHT" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE3D_MAX_HEIGHT) << endl;
            os << "CL_DEVICE_IMAGE3D_MAX_WIDTH" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE3D_MAX_WIDTH) << endl;
#ifdef CL_VERSION_1_2
            os << "CL_DEVICE_IMAGE_MAX_BUFFER_SIZE" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE) << endl;
            os << "CL_DEVICE_IMAGE_MAX_ARRAY_SIZE" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE) << endl;
            os << "CL_DEVICE_LINKER_AVAILABLE" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_LINKER_AVAILABLE) << endl;
#endif
            os << "CL_DEVICE_LOCAL_MEM_SIZE" << sep << context->getInfoWithDefaultOnError<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE) << endl;

            {
                os << "CL_DEVICE_LOCAL_MEM_TYPE" << sep;
                cl_device_local_mem_type memType = context->getInfoWithDefaultOnError<cl_device_local_mem_type>(CL_DEVICE_LOCAL_MEM_TYPE);
                switch(memType)
                {
                    case CL_LOCAL:
                        os << "CL_LOCAL";
                        break;
                    case CL_GLOBAL:
                        os << "CL_GLOBAL";
                        break;
                }
                os << endl;
            }

            os << "CL_DEVICE_MAX_CLOCK_FREQUENCY" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY) << endl;
            os << "CL_DEVICE_MAX_COMPUTE_UNITS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS) << endl;
            os << "CL_DEVICE_MAX_CONSTANT_ARGS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS) << endl;
            os << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" << sep << context->getInfoWithDefaultOnError<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE) << endl;
            os << "CL_DEVICE_MAX_MEM_ALLOC_SIZE" << sep << context->getInfoWithDefaultOnError<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) << endl;
            os << "CL_DEVICE_MAX_PARAMETER_SIZE" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_MAX_PARAMETER_SIZE) << endl;
            os << "CL_DEVICE_MAX_READ_IMAGE_ARGS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_READ_IMAGE_ARGS) << endl;
            os << "CL_DEVICE_MAX_SAMPLERS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_SAMPLERS) << endl;
            os << "CL_DEVICE_MAX_WORK_GROUP_SIZE" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE) << endl;

            {
                cl_uint dimensions = context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
                os << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS" << sep << dimensions << endl;

                os << "CL_DEVICE_MAX_WORK_ITEM_SIZES" << sep;
                size_t* sizes = (size_t*)context->getInfoWithDefaultOnError(CL_DEVICE_MAX_WORK_ITEM_SIZES);
                if(sizes != nullptr)
                {
                    copy(sizes, sizes + dimensions, ostream_iterator<size_t>(os, ","));
                    delete sizes;
                }
                os << endl;
            }

            os << "CL_DEVICE_MAX_WRITE_IMAGE_ARGS" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_WRITE_IMAGE_ARGS) << endl;
            os << "CL_DEVICE_MEM_BASE_ADDR_ALIGN" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MEM_BASE_ADDR_ALIGN) << endl;
            os << "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE) << endl;
            os << "CL_DEVICE_NAME" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_NAME) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_INT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE) << endl;
            os << "CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF) << endl;
            os << "CL_DEVICE_OPENCL_C_VERSION" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_OPENCL_C_VERSION) << endl;
			#ifdef CL_VERSION_1_2
            os << "CL_DEVICE_PARENT_DEVICE" << sep << context->getInfoWithDefaultOnError<cl_device_id>(CL_DEVICE_PARENT_DEVICE) << endl;
            os << "CL_DEVICE_PARTITION_MAX_SUB_DEVICES" << sep << context->getInfoWithDefaultOnError<cl_device_id>(CL_DEVICE_PARTITION_MAX_SUB_DEVICES) << endl;

			{
			os << "CL_DEVICE_PARTITION_PROPERTIES" << sep;
                cl_device_partition_property properties[] = context->getInfoWithDefaultOnError<cl_device_partition_property[]>(CL_DEVICE_PARTITION_PROPERTIES);
				while(*properties != 0) {
					switch(*properties) {
					case CL_DEVICE_PARTITION_EQUALLY:
					    os << "CL_DEVICE_PARTITION_EQUALLY";
					    break;
					case CL_DEVICE_PARTITION_BY_COUNTS:
					    os << "CL_DEVICE_PARTITION_BY_COUNTS";
					    break;
					case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
					    os << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN";
					    break;
					}
					properties++;
				}
            }

            {
                os << "CL_DEVICE_PARTITION_AFFINITY_DOMAIN" << sep;
                cl_device_affinity_domain flags = context->getInfoWithDefaultOnError<cl_device_affinity_domain>(CL_DEVICE_PARTITION_AFFINITY_DOMAIN);
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_NUMA)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_NUMA ";
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_L4_CACHE ";
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_L3_CACHE ";
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_L2_CACHE ";
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_L1_CACHE ";
                if(flags & CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE)
                    os << "CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE ";

                os << endl;
            }

			{
				os << "CL_DEVICE_PARTITION_TYPE" << sep;
                cl_device_partition_property properties[] = context->getInfoWithDefaultOnError<cl_device_partition_property[]>(CL_DEVICE_PARTITION_TYPE);
				while(*properties != 0) {
					switch(*properties) {
					case CL_DEVICE_PARTITION_EQUALLY:
					    os << "CL_DEVICE_PARTITION_EQUALLY";
					    break;
					case CL_DEVICE_PARTITION_BY_COUNTS:
					    os << "CL_DEVICE_PARTITION_BY_COUNTS";
					    break;
					case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
					    os << "CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN";
					    break;
					}
					properties++;
				}
            }
#endif

            os << "CL_DEVICE_PLATFORM" << sep << context->getInfoWithDefaultOnError<cl_platform_id>(CL_DEVICE_PLATFORM) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE) << endl;
            os << "CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF) << endl;
#ifdef CL_VERSION_1_2
            os << "CL_DEVICE_PRINTF_BUFFER_SIZE" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_PRINTF_BUFFER_SIZE) << endl;

            os << "CL_DEVICE_PREFERRED_INTEROP_USER_SYNC" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC) << endl;
#endif
            os << "CL_DEVICE_PROFILE" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_PROFILE) << endl;
            os << "CL_DEVICE_PROFILING_TIMER_RESOLUTION" << sep << context->getInfoWithDefaultOnError<size_t>(CL_DEVICE_PROFILING_TIMER_RESOLUTION) << endl;

            {
                os << "CL_DEVICE_QUEUE_PROPERTIES" << sep;
                cl_command_queue_properties properties = context->getInfoWithDefaultOnError<cl_command_queue_properties>(CL_DEVICE_QUEUE_PROPERTIES);
                if(properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
                    os << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
                if(properties & CL_QUEUE_PROFILING_ENABLE)
                    os << "CL_QUEUE_PROFILING_ENABLE ";
                os << endl;
            }
#ifdef CL_VERSION_1_2
			os << "CL_DEVICE_REFERENCE_COUNT" << sep << context->getInfoWithDefaultOnError<cl_bool>(CL_DEVICE_REFERENCE_COUNT) << endl;
#endif
            {
                os << "CL_DEVICE_SINGLE_FP_CONFIG" << sep;
                cl_device_fp_config flags = context->getInfoWithDefaultOnError<cl_device_fp_config>(CL_DEVICE_SINGLE_FP_CONFIG);
                if(flags & CL_FP_DENORM)
                    os << "CL_FP_DENORM ";
                if(flags & CL_FP_INF_NAN)
                    os << "CL_FP_INF_NAN ";
                if(flags & CL_FP_ROUND_TO_NEAREST)
                    os << "CL_FP_ROUND_TO_NEAREST ";
                if(flags & CL_FP_ROUND_TO_ZERO)
                    os << "CL_FP_ROUND_TO_ZERO ";
                if(flags & CL_FP_ROUND_TO_INF)
                    os << "CL_FP_ROUND_TO_INF ";
#ifdef CL_VERSION_1_2
				if(flags & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
					os << "CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT ";
#endif
                if(flags & CL_FP_FMA)
                    os << "CP_FP_FMA ";
                if(flags & CL_FP_SOFT_FLOAT)
                    os << "CL_FP_SOFT_FLOAT ";
                os << endl;
            }

            {
                os << "CL_DEVICE_TYPE" << sep;
                cl_device_type deviceType = context->getInfoWithDefaultOnError<cl_device_type>(CL_DEVICE_TYPE);
                switch(deviceType)
                {
                    case CL_DEVICE_TYPE_CPU:
                        os << "CL_DEVICE_TYPE_CPU";
                        break;
                    case CL_DEVICE_TYPE_GPU:
                        os << "CL_DEVICE_TYPE_GPU";
                        break;
                    case CL_DEVICE_TYPE_ACCELERATOR:
                        os << "CL_DEVICE_TYPE_ACCELERATOR";
                        break;
                    case CL_DEVICE_TYPE_DEFAULT:
                        os << "CL_DEVICE_TYPE_DEFAULT";
                        break;
#ifdef CL_VERSION_1_2
					case CL_DEVICE_TYPE_CUSTOM:
                        os << "CL_DEVICE_TYPE_CUSTOM";
                        break;
#endif
                }
                os << endl;
            }

            os << "CL_DEVICE_VENDOR" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_VENDOR) << endl;
            os << "CL_DEVICE_VENDOR_ID" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_VENDOR_ID) << endl;
            os << "CL_DEVICE_VERSION" << sep << context->getInfoWithDefaultOnError<string>(CL_DEVICE_VERSION) << endl;
            os << "CL_DRIVER_VERSION" << sep << context->getInfoWithDefaultOnError<string>(CL_DRIVER_VERSION) << endl;
			#ifdef CL_VERSION_1_2
            os << "CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT" << sep << context->getInfoWithDefaultOnError<cl_uint>(CL_DEVICE_MAX_ATOMIC_COUNTERS_EXT) << endl;
#endif
            os.close();

            cout << "DONE" << endl;
        }
};

#endif // STATSWRITER_H

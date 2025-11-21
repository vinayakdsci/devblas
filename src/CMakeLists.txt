add_library(devblas SHARED
	sgemm.cc
)

target_include_directories(devblas PUBLIC ../include/)
set_target_properties(devblas PROPERTIES
    VISIBILITY_INLINES_HIDDEN true
)

install(TARGETS devblas DESTINATION devblas)



#ifndef MRCV_EXPORT_H
#define MRCV_EXPORT_H

#ifdef MRCV_STATIC_DEFINE
#  define MRCV_EXPORT
#  define MRCV_NO_EXPORT
#else
#  ifndef MRCV_EXPORT
#    ifdef mrcv_EXPORTS
        /* We are building this library */
#      define MRCV_EXPORT 
#    else
        /* We are using this library */
#      define MRCV_EXPORT 
#    endif
#  endif

#  ifndef MRCV_NO_EXPORT
#    define MRCV_NO_EXPORT 
#  endif
#endif

#ifndef MRCV_DEPRECATED
#  define MRCV_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef MRCV_DEPRECATED_EXPORT
#  define MRCV_DEPRECATED_EXPORT MRCV_EXPORT MRCV_DEPRECATED
#endif

#ifndef MRCV_DEPRECATED_NO_EXPORT
#  define MRCV_DEPRECATED_NO_EXPORT MRCV_NO_EXPORT MRCV_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef MRCV_NO_DEPRECATED
#    define MRCV_NO_DEPRECATED
#  endif
#endif

#endif /* MRCV_EXPORT_H */

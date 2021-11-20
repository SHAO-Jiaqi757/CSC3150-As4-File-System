#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
//#define __device__
//#define __managed__

typedef unsigned char uchar;
typedef uint32_t u32;
typedef uint16_t u16;
#define G_WRITE 1
#define G_READ 0
#define LS_D 0
#define LS_S 1
#define RM 2
#define ERROR UINT32_MAX
struct SuperBlock
{
    int free_block_count; // how many free block
    u16 free_block_start; // the first start free block number;
    int file_num = 0;     // how many files in the storge
};
#pragma pack(1)
struct FCB
{
    u32 modified_time; // 4 bytes
    u32 create_time;   // 4 bytes
    u16 file_size;     // 2 bytes
    u16 start_block;
    char filename[20];
};
#pragma pack()

struct FileSystem
{
    uchar *volume;
    int SUPERBLOCK_SIZE;
    int FCB_SIZE;
    int FCB_ENTRIES;
    int STORAGE_SIZE;
    int STORAGE_BLOCK_SIZE;
    int MAX_FILENAME_SIZE;
    int MAX_FILE_NUM;
    int MAX_FILE_SIZE;
    int FILE_BASE_ADDRESS;

    SuperBlock *superBlock_ptr;
    struct FCB *FCB_arr;
    uchar *fileContent_ptr;
};

__device__ void init_volume(FileSystem *fs);
__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
                        int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
                        int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE,
                        int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS);
__device__ char *my_strcpy(char *dst, const char *src);
__device__ int my_strcmp(char *s1, char *s2);
__device__ void compact(FileSystem *fs, u16 block_num, u16 start_block);
__device__ void remove_file(FileSystem *fs, int fp);
__device__ void sort_file(FileSystem *fs, int op);
__device__ void swap_file(FileSystem *fs, int fp1, int fp2);
__device__ u32 fs_open(FileSystem *fs, char *s, int op);
__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp);
__device__ u32 fs_write(FileSystem *fs, uchar *input, u32 size, u32 fp);
__device__ void fs_gsys(FileSystem *fs, int op);
__device__ void fs_gsys(FileSystem *fs, int op, char *s);

#endif

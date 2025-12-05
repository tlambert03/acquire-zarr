"""
Example demonstrating the write_group_metadata option.

This example shows how to write only array nodes (and their zarr.json files)
without writing group metadata (zarr.json files for parent directories).

This is useful when:
- Writing to a pre-existing Zarr store where group metadata is already managed
- Integrating with external systems that handle group metadata separately
- Minimizing redundant metadata writes in performance-critical applications
"""

import numpy as np
import acquire_zarr


def main():
    # Create stream settings
    settings = acquire_zarr.StreamSettings(
        store_path="test_array_only.zarr",
        overwrite=True,
        # Set write_group_metadata=False to skip writing group zarr.json files
        # Only array zarr.json files will be written
        write_group_metadata=False,
    )

    # Configure an array with multiscale (normally creates a group)
    x = acquire_zarr.Dimension(
        name="x",
        kind=acquire_zarr.DimensionType.SPACE,
        array_size_px=1920,
        chunk_size_px=1920,
        shard_size_chunks=1,
    )
    y = acquire_zarr.Dimension(
        name="y",
        kind=acquire_zarr.DimensionType.SPACE,
        array_size_px=1080,
        chunk_size_px=1080,
        shard_size_chunks=1,
    )
    t = acquire_zarr.Dimension(
        name="t",
        kind=acquire_zarr.DimensionType.TIME,
        array_size_px=0,  # 0 means unlimited
        chunk_size_px=1,
        shard_size_chunks=1,
    )

    settings.arrays = [
        acquire_zarr.ArraySettings(
            output_key="my_array",
            dimensions=[t, y, x],
            data_type=acquire_zarr.DataType.UINT16,
            # With downsampling_method set, this would normally create:
            # - my_array/zarr.json (group metadata with OME multiscales)
            # - my_array/0/zarr.json (array metadata)
            # - my_array/1/zarr.json (array metadata for downsampled)
            # etc.
            #
            # With write_group_metadata=False, only the array zarr.json
            # files (my_array/0/zarr.json, my_array/1/zarr.json) are written.
            # The group zarr.json (my_array/zarr.json) is NOT written.
            downsampling_method=acquire_zarr.DownsamplingMethod.MEAN,
        )
    ]

    # Create the stream
    stream = acquire_zarr.ZarrStream(settings)

    # Write some frames
    for i in range(5):
        frame = np.random.randint(
            0, 65535, (1080, 1920), dtype=np.uint16
        )
        stream.append(frame)

    # Close the stream
    stream.close()

    print("Stream complete!")
    print("With write_group_metadata=False:")
    print("  - Array data and array zarr.json files are written")
    print("  - Group zarr.json files are NOT written")


if __name__ == "__main__":
    main()

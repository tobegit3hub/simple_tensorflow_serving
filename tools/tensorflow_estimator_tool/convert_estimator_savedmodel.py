#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib import graph_editor


def main():
  with tf.Session(graph=tf.Graph()) as sess:

    tag_name = "serve"
    signature_name = "serving_default"
    input_model_path = "./savedmodel/1531987620"
    output_model_path = "./converted_savedmodel/1531987620"

    graph = tf.get_default_graph()
    meta_graph_def = tf.saved_model.loader.load(sess, [tag_name],
                                                input_model_path)
    signature_def = meta_graph_def.signature_def[signature_name]

    old_input_tensor = graph.get_tensor_by_name(
        signature_def.inputs.values()[0].name)
    print "Old model input: ", old_input_tensor.dtype, old_input_tensor.shape

    new_base64_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    new_input_tensor = tf.map_fn(tf.decode_base64, new_base64_placeholder)
    print "New model input: ", old_input_tensor.dtype, old_input_tensor.shape

    output_keys = [_.name for _ in signature_def.outputs.values()]
    old_output_tensors = [graph.get_tensor_by_name(_) for _ in output_keys]

    new_output_tensors = graph_editor.graph_replace(
        target_ts=old_output_tensors,
        replacement_ts={old_input_tensor: new_input_tensor})
    new_output_tensor_infos = [
        tf.saved_model.utils.build_tensor_info(_) for _ in new_output_tensors
    ]

    new_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={
            signature_def.inputs.keys()[0]:
            tf.saved_model.utils.build_tensor_info(new_base64_placeholder)
        },
        outputs=dict(zip(output_keys, new_output_tensor_infos)),
        method_name=signature_def.method_name)
    builder = tf.saved_model.builder.SavedModelBuilder(output_model_path)
    builder.add_meta_graph_and_variables(
        sess,
        [tag_name],
        clear_devices=True,
        signature_def_map={
            signature_name: new_signature,
        },
        main_op=None,  # avoid duplicate
        legacy_init_op=None  # avoid duplicate
    )
    builder.save()


if __name__ == "__main__":
  main()

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <ctime>
#include <cmath>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define DLLEXPORT extern "C" __declspec(dllexport)
#else
#define DLLEXPORT
#endif

DLLEXPORT typedef struct MLP_m {
    std::vector<std::vector<std::vector<float>>> W;
    std::vector<int> d;
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> deltas;

    void forward_pass(const float *sample_inputs, bool is_classification) {
        unsigned long L = d.size() - 1;
        for (int j = 1; j < (d[0] + 1); j++) {
            X[0][j] = sample_inputs[j - 1];
        }
        for (int l = 1; l < (L+1); l++) {
            for (int j = 1; j < d[l] + 1; j++) {
                float sum_result = 0.0;
                for (int i = 0; i < (d[l - 1] + 1); i++) {
                    sum_result += W[l][i][j] * X[l - 1][i];
                }
                X[l][j] = sum_result;
                if (is_classification || l < L) {
                    X[l][j] = float(tanh(X[l][j]));
                }
            }
        }
    }

    static float *get_sample(const float *tab, int first_index, int last_index) {
        int len = last_index - first_index;
        auto *sample = new float[len];
        for (int i = 0; i < len; i++) {
            sample[i] = tab[first_index + i];
        }
        return sample;
    }

    void train_stochastic_gradient_backpropagation(float *flattened_dataset_inputs, int flattened_dataset_inputs_len, float *flattened_dataset_expected_outputs, bool is_classification, float alpha=0.001, int iterations_count=100000) {
        srand(time(nullptr));
        rand();
        int input_dim = d[0];
        unsigned long last_index = d.size() - 1;
        int output_dim = d[last_index];
        int sample_count = int(floor(double(flattened_dataset_inputs_len) / double(input_dim)));
        unsigned long L = d.size() - 1;
        for (int it = 0; it < iterations_count; it++) {
            int k = rand()%(sample_count);
            float *sample_input = get_sample(flattened_dataset_inputs, (k * input_dim), ((k+1) * input_dim));
            float *sample_expected_output = get_sample(flattened_dataset_expected_outputs, (k * output_dim), ((k+1) * output_dim));
            forward_pass(sample_input, is_classification);
            for (int j = 1; j < d[L] + 1; j++) {
                deltas[L][j]=(X[L][j] - sample_expected_output[j-1]);
                if (is_classification) {
                    deltas[L][j] *= (1 - X[L][j] * X[L][j]);
                }
            }
            for (unsigned long l = L; l > 0 ; l--) {
                for (int i = 1; i < d[l - 1] + 1; i++) {
                    float sum_result = 0.0;
                    for (int j = 1; j < d[l] + 1; j++) {
                        sum_result += W[l][i][j] * deltas[l][j];
                    }
                    deltas[l - 1][i] = (1 - X[l - 1][i] * X[l - 1][i]) * sum_result;
                }
            }
            for (int l = 1; l < L + 1; l++) {
                for (int i = 0; i < d[l - 1] + 1; i++) {
                    for (int j = 1; j < d[l] + 1; j++) {
                        W[l][i][j] -= alpha * X[l - 1][i] * deltas[l][j];
                    }
                }
            }
        }
    }
}MLP;

DLLEXPORT MLP *create_mlp_model(int *npl, int npl_size) {
    srand(time(nullptr));
    rand();
    MLP *model = new MLP[1];
    for (int l = 0; l < npl_size; l++) {
        model->W.emplace_back(0);
        if (l != 0) {
            for (int i = 0; i < (npl[l - 1] + 1); i++) {
                model->W[l].push_back(std::vector<float>(npl[l] + 1));
                for (int j = 0; j < npl[l] + 1; j++) {
                    model->W[l][i][j] = float(rand() % 3 - 1);
                }
            }
        }
    }
    for (int i = 0; i < npl_size; i++) {
        model->d.push_back(npl[i]);
    }
    for (int l = 0; l < npl_size; l++) {
        model->X.emplace_back(0);
        for (int j = 0; j < (npl[l] + 1); j++) {
            if (j == 0) {
                model->X[l].push_back(1.0);
            } else {
                model->X[l].push_back(0.0);
            }
        }
    }
    for (int l = 0; l < npl_size; l++) {
        model->deltas.emplace_back(0);
        for (int j = 0; j < (npl[l] + 1); j++) {
            model->deltas[l].push_back(0.0);
        }
    }
    return model;
}

DLLEXPORT float *predict_mlp_model_regression(MLP *model, float *sample_inputs) {
    model->forward_pass(sample_inputs, false);
    unsigned long last_index = (model->X.size()) - 1;
    unsigned long size = (model->X[last_index].size()) - 1;
    auto convertedVector = new float[size];
    for (int i = 0; i < size; i++) {
        convertedVector[i] = model->X[last_index][i + 1];
    }
    return convertedVector;
}

DLLEXPORT float *predict_mlp_model_classification(MLP *model, float *sample_inputs) {
    model->forward_pass(sample_inputs, true);
    unsigned long last_index = (model->X.size()) - 1;
    unsigned long size = (model->X[last_index].size()) - 1 ;
    auto convertedVector = new float[size];
    for (int i = 0; i < size; i++) {
        convertedVector[i] = model->X[last_index][i + 1];
    }
    return convertedVector;
}

DLLEXPORT void train_classification_stochastic_gradient_backpropagation_mlp_model(MLP *model, float *flattened_dataset_inputs, int flattened_dataset_inputs_size, float *flattened_dataset_expected_outputs, float alpha = 0.001, int iterations_count = 100000) {
    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_dataset_inputs_size, flattened_dataset_expected_outputs, true, alpha, iterations_count);
}

DLLEXPORT void train_regression_stochastic_gradient_backpropagation_mlp_model(MLP *model, float *flattened_dataset_inputs, int flattened_dataset_inputs_size, float *flattened_dataset_expected_outputs, float alpha = 0.001, int iterations_count = 100000) {
    model->train_stochastic_gradient_backpropagation(flattened_dataset_inputs, flattened_dataset_inputs_size,flattened_dataset_expected_outputs, false, alpha, iterations_count);
}

DLLEXPORT int getXSize(MLP *model) {
    return int((model->X[(model->X.size()) - 1].size()) - 1);
}